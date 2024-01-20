import React from 'react';
// import './App.css'; // Importing CSS for styling
import 'bootstrap/dist/css/bootstrap.min.css';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';


import Card from 'react-bootstrap/Card';
import ListGroup from 'react-bootstrap/ListGroup';

const VideoItem: React.FC<{ image: string; title: string; description: string }> = ({ image, title, description }) => (
    <Card style={{ background: "#1f1f1f", width: '12rem', borderRadius: "0 0 .83333rem .83333rem" }}>
      <Card.Img variant="top" width="12rem" src={image}  />
      <Card.Body style={{
        background: 'linear-gradient(92deg,#332b28 0%,#1a2933 100%)',
        borderRadius: "0 0 .83333rem .83333rem"
      }}>
      <Card.Text style={{
        color: '#fff',
        background: 'linear-gradient(90deg,#ffd9c9 0%,#83cbff 100%)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}>{description}</Card.Text>
      </Card.Body>
    </Card>
  );

import { useEffect, useState } from 'react';

const App = () => {
  interface Video {
    image: string;
    title: string;
    description: string;
  }

  const [videos, setVideos] = useState<Video[]>([]);

  useEffect(() => {
    fetch('outputs/exp.json')
      .then(response => {
        console.log(response);
        return response.json();
      })
      .then(data => setVideos(data.results));
  }, []);

  return (
    <>
      <Row>
          <ListGroup className="collections" >
            <Row className="g-4" style={{paddingLeft: "3%"}}>
              {videos.map((video, index) => (
                <Col key={index}>
                  <VideoItem image={video.image} title={video.title} description={video.description} />
                </Col>
              ))}
            </Row>
          </ListGroup>
      </Row>
    </>
  );
};


export default App;
