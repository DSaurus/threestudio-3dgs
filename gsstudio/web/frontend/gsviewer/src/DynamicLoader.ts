import * as SPLAT from "gsplat";

class PLYDynamicLoader {
    static SH_C0 = 0.28209479177387814;

    static etag: string | null = null;
    static lastModified: string | null = null;

    static async LoadAsync(
        url: string,
        scene: SPLAT.Scene,
        onProgress?: (progress: number) => void,
        format: string = "",
        useCache: boolean = false,
    ): Promise<SPLAT.Splat> {
        let headers = new Headers();
        if (PLYDynamicLoader.etag) {
            headers.append('If-None-Match', PLYDynamicLoader.etag);
        }
        if (PLYDynamicLoader.lastModified) {
            headers.append('If-Modified-Since', PLYDynamicLoader.lastModified);
        }
        const req = await fetch(url, {
            mode: "cors",
            credentials: "omit",
            cache: useCache ? "force-cache" : "default",
            headers: headers,
        });

        console.log("req", req.status);
        if (req.status === 304) {
            // The file hasn't changed
            return new SPLAT.Splat();
        }
        if (req.status != 200) {
            throw new Error(req.status + " Unable to load " + req.url);
        }
        PLYDynamicLoader.etag = req.headers.get('ETag');
        PLYDynamicLoader.lastModified = req.headers.get('Last-Modified');


        const reader = req.body!.getReader();
        const contentLength = parseInt(req.headers.get("content-length") as string);
        const plyData = new Uint8Array(contentLength);

        let bytesRead = 0;

        // eslint-disable-next-line no-constant-condition
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            plyData.set(value, bytesRead);
            bytesRead += value.length;

            onProgress?.(bytesRead / contentLength);
        }

        if (plyData[0] !== 112 || plyData[1] !== 108 || plyData[2] !== 121 || plyData[3] !== 10) {
            throw new Error("Invalid PLY file");
        }

        const buffer = new Uint8Array(this._ParsePLYBuffer(plyData.buffer, format));
        const data = SPLAT.SplatData.Deserialize(buffer);
        const splat = new SPLAT.Splat(data);
        scene.addObject(splat);
        while (scene.objects.length > 1) {
            scene.removeObject(scene.objects[0]);
        }
        return splat;
    }

    static async LoadFromFileAsync(
        file: File,
        scene: SPLAT.Scene,
        onProgress?: (progress: number) => void,
        format: string = "",
    ): Promise<SPLAT.Splat> {
        const reader = new FileReader();
        let splat = new SPLAT.Splat();
        reader.onload = (e) => {
            const buffer = new Uint8Array(this._ParsePLYBuffer(e.target!.result as ArrayBuffer, format));
            const data = SPLAT.SplatData.Deserialize(buffer);
            splat = new SPLAT.Splat(data);
            scene.addObject(splat);
        };
        reader.onprogress = (e) => {
            onProgress?.(e.loaded / e.total);
        };
        reader.readAsArrayBuffer(file);
        await new Promise<void>((resolve) => {
            reader.onloadend = () => {
                resolve();
            };
        });
        return splat;
    }

    private static _ParsePLYBuffer(inputBuffer: ArrayBuffer, format: string): ArrayBuffer {
        type PlyProperty = {
            name: string;
            type: string;
            offset: number;
        };

        const ubuf = new Uint8Array(inputBuffer);
        const headerText = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
        const header_end = "end_header\n";
        const header_end_index = headerText.indexOf(header_end);
        if (header_end_index < 0) throw new Error("Unable to read .ply file header");

        const vertexCount = parseInt(/element vertex (\d+)\n/.exec(headerText)![1]);

        let rowOffset = 0;
        const offsets: Record<string, number> = {
            double: 8,
            int: 4,
            uint: 4,
            float: 4,
            short: 2,
            ushort: 2,
            uchar: 1,
        };

        const properties: PlyProperty[] = [];
        for (const prop of headerText
            .slice(0, header_end_index)
            .split("\n")
            .filter((k) => k.startsWith("property "))) {
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const [_p, type, name] = prop.split(" ");
            properties.push({ name, type, offset: rowOffset });
            if (!offsets[type]) throw new Error(`Unsupported property type: ${type}`);
            rowOffset += offsets[type];
        }

        const dataView = new DataView(inputBuffer, header_end_index + header_end.length);
        const buffer = new ArrayBuffer(SPLAT.SplatData.RowLength * vertexCount);

        const q_polycam = SPLAT.Quaternion.FromEuler(new SPLAT.Vector3(-Math.PI / 2, 0, 0));

        for (let i = 0; i < vertexCount; i++) {
            const position = new Float32Array(buffer, i * SPLAT.SplatData.RowLength, 3);
            const scale = new Float32Array(buffer, i * SPLAT.SplatData.RowLength + 12, 3);
            const rgba = new Uint8ClampedArray(buffer, i * SPLAT.SplatData.RowLength + 24, 4);
            const rot = new Uint8ClampedArray(buffer, i * SPLAT.SplatData.RowLength + 28, 4);

            let r0: number = 255;
            let r1: number = 0;
            let r2: number = 0;
            let r3: number = 0;

            properties.forEach((property) => {
                let value;
                switch (property.type) {
                    case "float":
                        value = dataView.getFloat32(property.offset + i * rowOffset, true);
                        break;
                    case "int":
                        value = dataView.getInt32(property.offset + i * rowOffset, true);
                        break;
                    default:
                        throw new Error(`Unsupported property type: ${property.type}`);
                }

                switch (property.name) {
                    case "x":
                        position[0] = value;
                        break;
                    case "y":
                        position[1] = value;
                        break;
                    case "z":
                        position[2] = value;
                        break;
                    case "scale_0":
                        scale[0] = Math.exp(value);
                        break;
                    case "scale_1":
                        scale[1] = Math.exp(value);
                        break;
                    case "scale_2":
                        scale[2] = Math.exp(value);
                        break;
                    case "red":
                        rgba[0] = value;
                        break;
                    case "green":
                        rgba[1] = value;
                        break;
                    case "blue":
                        rgba[2] = value;
                        break;
                    case "f_dc_0":
                        rgba[0] = (0.5 + this.SH_C0 * value) * 255;
                        break;
                    case "f_dc_1":
                        rgba[1] = (0.5 + this.SH_C0 * value) * 255;
                        break;
                    case "f_dc_2":
                        rgba[2] = (0.5 + this.SH_C0 * value) * 255;
                        break;
                    case "f_dc_3":
                        rgba[3] = (0.5 + this.SH_C0 * value) * 255;
                        break;
                    case "opacity":
                        rgba[3] = (1 / (1 + Math.exp(-value))) * 255;
                        break;
                    case "rot_0":
                        r0 = value;
                        break;
                    case "rot_1":
                        r1 = value;
                        break;
                    case "rot_2":
                        r2 = value;
                        break;
                    case "rot_3":
                        r3 = value;
                        break;
                }
            });

            let q = new SPLAT.Quaternion(r1, r2, r3, r0);

            switch (format) {
                case "polycam": {
                    // const temp = position[1];
                    // position[1] = -position[2];
                    // position[2] = temp;
                    // position[1] = -position[1];
                    // position[0] = -position[0];
                    // position[2] = -position[2];
                    // q = q_polycam.multiply(q);
                    break;
                }
                case "":
                    break;
                default:
                    throw new Error(`Unsupported format: ${format}`);
            }

            q = q.normalize();
            rot[0] = q.w * 128 + 128;
            rot[1] = q.x * 128 + 128;
            rot[2] = q.y * 128 + 128;
            rot[3] = q.z * 128 + 128;
        }

        return buffer;
    }
}

export { PLYDynamicLoader };
