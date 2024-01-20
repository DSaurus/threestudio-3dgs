import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
// import './index.css'
// interface Video {
//   image: string;
//   title: string;
//   description: string;
// }

// const [videos, setVideos] = useState<Video[]>([]);

// useEffect(() => {
//   fetch('outputs/exp.json')
//     .then(response => {
//       console.log(response);
//       return response.json();
//     })
//     .then(data => setVideos(data.results));
// }, []);

// return (
//   <>
//     <Row>
//         <ListGroup className="collections" >
//           <Row className="g-4" style={{paddingLeft: "3%"}}>
//             {videos.map((video, index) => (
//               <Col key={index}>
//                 <VideoItem image={video.image} title={video.title} description={video.description} />
//               </Col>
//             ))}
//           </Row>
//         </ListGroup>
//     </Row>
//   </>
// );
// ReactDOM.createRoot(document.getElementById('root')!).render(
//   <React.StrictMode>
//     <App />
//   </React.StrictMode>,
// )
