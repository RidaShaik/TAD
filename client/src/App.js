import React, {useState, useEffect, useRef} from 'react'
import './App.css'

import logo from './logo.png';

function App() {

    const videoRef = useRef(null);
    const [currentAnnotation, setCurrentAnnotation] = useState("");

/*const handleTimeUpdate = () => {
    if (videoRef.current) {
        const currentTimeDec = videoRef.current.currentTime.toFixed(1);
        const currentTimeInt = Math.floor(videoRef.current.currentTime)
        setCurrentAnnotation(`GT[${currentTimeInt}] start:${currentTimeDec}`);
      }
};*/

    const handleTimeUpdate = () => {
        if (videoRef.current) {
            const currentTime = videoRef.current.currentTime;
            const fps = 30; // If your videos are 30 FPS; adjust if needed
            const frameIndex = Math.floor(currentTime * fps);

            if (frameIndex >= 0 && frameIndex < frameAnnotations.length) {
                const labels = frameAnnotations[frameIndex];

                if (labels && labels.length > 0) {
                    const timestamp = currentTime.toFixed(1); // example: "3.4"
                    const labelText = labels.join(", ");
                    setCurrentAnnotation(`Time: ${timestamp}s\nLabels: ${labelText}`);
                } else {
                    const timestamp = currentTime.toFixed(1);
                    setCurrentAnnotation(`Time: ${timestamp}s\n(No action detected)`);
                }
            } else {
                setCurrentAnnotation("(Waiting for video to start...)");
            }
        }
    };

    const [videoFile, setVideoFile] = useState(null);
    const [videoURL, setVideoURL] = useState('');
    const [videos, setVideos] = useState([]);
    const [selectedVideo, setSelectedVideo] = useState("");
    const [currentPage, setCurrentPage] = useState("home");

    const [newsArticles, setNewsArticles] = useState([]);
    const [playDetectionVideo, setPlayDetectionVideo] = useState(false);

    const [frameAnnotations, setFrameAnnotations] = useState([]); // EDIT HERE 2


    useEffect(() => {
        fetchVideos();
    }, []);

    useEffect(() => {
        fetchSportsNews();
    }, []);

    const fetchVideos = () => {
        fetch("/videos")
            .then(res => res.json())
            .then(data => {
                console.log("Fetched Videos", data);
                if (Array.isArray(data.videos)) {
                    setVideos(data.videos);
                }
                else {
                    setVideos([]);
                }
            })
        .catch(error => {
            console.error("Error fetching Videos", error);
            setVideos([]);
        });
    };

    const fetchSportsNews = async () => {
        try {
            const response = await fetch(`https://newsapi.org/v2/top-headlines?category=sports&language=en&pageSize=5&apiKey=acda77d8110a49ef8f1dfcfa849d0697`);
            const data = await response.json();
            setNewsArticles(data.articles);
        }
        catch (error) {
            console.error("Error fetching SportsNews", error);
        }
    }

    const handleVideoUpload = async (event) => {
        const file = event.target.files[0];

        if (!file)
            return;

        setVideoFile(file);

        const formData = new FormData();
        formData.append('video', file);

        const response = await fetch("/upload", {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            alert("Video uploaded successfully!");
            fetchVideos();
        } else {
            alert("Video upload failed.");
        }
    };

    /*const handleVideoSelect = (event) => {
        const filename = event.target.value;
        setSelectedVideo(filename);
        setVideoURL(`/videos/${filename}`);
    }*/
    const handleVideoSelect = async (filename) => {
        setSelectedVideo(filename);
        setCurrentPage('detection');


        setVideoURL(`/videos/${filename}`);

        try {
            const response = await fetch("/inference", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    video_name: filename,

                })
            });

            if (response.ok) {
                const result = await response.json();
                const annotatedName = result.outputvideo || `annotated${filename}`;
                setVideoURL(`/videos/${annotatedName}`);

                if (result.frame_labels) { // EDIT HERE 2 BEGIN
                    setFrameAnnotations(result.frame_labels);
                }
                if (result.frame_labels) {
                    console.log("Frame Annotations received:", result.frame_labels);
                    setFrameAnnotations(result.frame_labels);
                } // EDIT HERE END

            } else {
                console.error("Inference failed");
            }
        } catch (error) {
            console.error("Inference error:", error);
        }
    };

    const getRandomVideos = () => {
        if (videos.length <= 5) return videos;
        const shuffled = [...videos].sort(() => 0.5 - Math.random());
        return shuffled.slice(0, 5);
    };

    return (
      <div className="app-container">
        <header classname="header">
          <div className="header-content">
            <img src={logo} alt="logo" className="logo" />
            <h1 className="home-header">Sports Action Detection</h1>
            <nav className="nav-bar">
              <button onClick={() => setCurrentPage('home')}>Home</button>
              <button onClick={() => setCurrentPage('clips')}>Clips</button>
              <button onClick={() => setCurrentPage('detection')}>
                Detection
              </button>
              <button onClick={() => setCurrentPage('upload')}>Upload</button>
            </nav>
          </div>
        </header>

        <br />
        <br />
        <br />

        <div classname="content">
          {currentPage === 'home' && (
            <div>
              <div className="video-container">
                <video autoPlay loop muted>
                  <source src="./homeVid.mp4" type="video/mp4" />
                  Vid Not Supported
                </video>
              </div>
              <div className="home-content">
                <h2 className="home-header">Welcome to the Video Analysis App</h2>
                <p>
                  This application allows you to upload videos and analyze them
                  using the ML model
                </p>
                <p>Navigate to the  <button className="home-button" onClick={() => setCurrentPage('clips')}><u>Clips</u></button> tab to use the ML model on existing videos within the web application</p>
                <p>Navigate to the  <button className="home-button" onClick={() => setCurrentPage('upload')}><u>Upload</u></button> tab to upload your own videos for analysis</p>
                <p>Navigate to the  <button className="home-button" onClick={() => setCurrentPage('detection')}><u>Detection</u></button> to view the ML results on your specified clip</p>
              </div>
            </div>
          )}

          {currentPage === 'clips' && (
            <div className="clips-container">
              {videos.length === 0 ? (
                <p>No videos uploaded yet.</p>
              ) : (
                <div className="video-grid">
                  {videos.map((video, index) => (
                    <div
                      className="video-card"
                      key={index}
                      onClick={() => {
                        setSelectedVideo(video);
                        setVideoURL(`/videos/${video}`);

                        handleVideoSelect(video);
                      }}
                    >
                      <video
                        width="320"
                        height="180"
                        className="clip-video"
                        muted
                        preload="metadata"
                      >
                        <source src={`/videos/${video}`} type="video/mp4" />
                        Your browser does not support the video tag.
                      </video>
                      <p className="video-title">{video}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {currentPage === 'detection' && (
            <div className="detection-container">
              {videoURL ? (
                <div className="detection-content">

                    {!playDetectionVideo && (
                        <button
                            className="begin-button"
                            onClick={() => {
                                setPlayDetectionVideo(true);
                            }}
                        >
                            Begin
                        </button>
                    )}

                    {playDetectionVideo && (
                        <video
                            className="detection-video"
                            src={videoURL}
                            controls
                            muted
                            ref={videoRef}
                            onTimeUpdate={handleTimeUpdate}
                            autoPlay
                        />
                    )}
                  <div className="analysis-box">
                    <h3>TAD Analysis</h3>
                    <p>Results from the ML model will appear here</p>
                    {currentAnnotation && (
                  <div className="annotation-box">
                    {currentAnnotation}
                  </div>
                    )}
                  </div>
                </div>

              ) : (
                <p>No Video Selected, Please select from Clips tab</p>
              )}
                <div className="highlight-reel">
                    <h2>Highlight Reel</h2>
                    <div className="highlight-grid">
                        {getRandomVideos().map((video, index) => (
                            <video
                                key={index}
                                className="highlight-video"
                                muted
                                loop
                                autoPlay
                                preload="metadata"
                            >
                                <source src={`/videos/${video}`} type="video/mp4" />
                                Your browser does not support the video tag.
                            </video>
                        ))}
                    </div>
                </div>
            </div>
          )}

          {currentPage === 'upload' && (
            <div className="upload-container">
              <div className="upload-content">
                <label htmlFor="upload-input" className="upload-button">
                  Upload Video
                </label>
                <input
                  id="upload-input"
                  type="file"
                  accept="video/*"
                  onChange={handleVideoUpload}
                  style={{ display: 'none' }}
                />

                  <div className="news-section">
                      <h2>Latest Sports News</h2>
                      {newsArticles.length > 0 ? (
                          <div className="news-grid">
                              {newsArticles.map((article, index) => (
                                  <div className="news-card" key={index}>
                                      {article.urlToImage && (
                                          <img src={article.urlToImage} alt="News" className="news-image" />
                                      )}
                                      <a href={article.url} target="_blank" rel="noopener noreferrer" className="news-card-title">
                                          {article.title}
                                      </a>
                                      <p className="news-card-description">{article.description}</p>
                                  </div>
                              ))}
                          </div>
                      ) : (
                          <p>Loading news...</p>
                      )}
                  </div>
              </div>
            </div>
          )}
        </div>
          <div className="footer-spacer"></div>
          <footer className="footer">
              <div className="footer-content">
                  <div className="footer-section">
                      <h4>Contact</h4>
                      <p>Email: sportsactiondetection@ufl.edu</p>
                      <p>Phone: (123)456-7890</p>
                  </div>
                  <div className="footer-section">
                      <h4>Follow Us</h4>
                      <div className="social-icons">
                          <a href="https://twitter.com" target="_blank" rel="noopener noreferrer">Twitter</a>
                          <a href="https://facebook.com" target="_blank" rel="noopener noreferrer">Facebook</a>
                          <a href="https://instagram.com" target="_blank" rel="noopener noreferrer">Instagram</a>
                      </div>
                  </div>
                  <div className="footer-section">
                      <h4>© 2025 Sports Action AI</h4>
                      <p>All rights reserved</p>
                  </div>
              </div>
          </footer>
      </div>
    );
}

export default App;