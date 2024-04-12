import React, { useState, useEffect } from "react";

import "../index.css";
// import Topbar from "./Topbar";
import Topbar from "../Topbar";
// import back2 from "./background/back2.jpg"
import back2 from "../background/back2.jpg";
import Result from "./Result";
import Loader from "./Loader";
import LeftBar from "./leftbar";
// BHushan

const ImageCaptionGenerator = () => {
  const [selectedFile, setSelectedFile] = useState("");
  const [selectedModel, setSelectedModel] = useState("LSTM");
  const [preview, setPreview] = useState("");
  const [bool, setBool] = useState(false);
  const [testImage, setTestImage] = useState(null);
  const [name, setName] = useState("");
  const handleTestImage = (data) => {
    setTestImage(data);
  };

  const handleImageChange = (event) => {
    // const img = event.target.files[0].name;
    const img = event.target.files[0];

    setSelectedFile(img);
  };

  const handleSelect = (value) => {
    setSelectedModel(value);
  };

  const handleGenerateCaption = async (event) => {
    if (testImage) {
      const response = await fetch(testImage);
      const blob = await response.blob();
      const file = new File([blob], "test.jpg", { type: blob.type });
      setSelectedFile(file);
      setBool(true);
    } else if (selectedFile) {
      setBool(true);
    } else {
      window.alert("Select image first");
    }
  };

  const fetchUser = async () => {
    const url = `http://localhost:8000/fetchnotes`;
    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        token: localStorage.getItem("token"),
      },
    });

    const json = await response.json();
    setName(json.firstname);
  };

  useEffect(() => {
    if (localStorage.getItem("token")) {
      fetchUser();
    }
  }, []);

  return (
    <>
      <div>
        {!bool && (
          <div
            className="divtop"
            style={{
              backgroundImage: `url(${back2})`,
              backgroundRepeat: "no-repeat",
              backgroundSize: "contain",
              height: 770,
              width: "100%",
              overflowY: "hidden",
            }}
          >
            <Topbar />
            <div className="div1">
              <div className="rightbar">
                {localStorage.getItem("token") ? (
                  <h1 className="heading">Hello {name}</h1>
                ) : (
                  <h1 className="heading">Welcome to NextWorld</h1>
                )}

                <h5 style={{ color: "black", fontSize: "16px" }}>
                  Let Images Speak <br />
                  Upload an Image to Generate Captivating Captions!
                </h5>

                <input
                  type="file"
                  style={{ color: "black" }}
                  onChange={handleImageChange}
                />

                <div className="imgdiv">
                  {preview && (
                    <img className="imgcss" src={preview} alt="image" />
                  )}
                </div>

                <div
                  style={{
                    color: "black",
                    fontSize: "16px",
                    marginTop: "10px",
                  }}
                >
                  <h3>Select any one Model</h3>
                  <div>
                    <label>
                      <input
                        type="radio"
                        name="valueSelector"
                        value="LSTM"
                        checked={selectedModel === "LSTM"}
                        onChange={() => handleSelect("LSTM")}
                      />
                      LSTM model
                    </label>

                    <label style={{ marginLeft: "15px" }}>
                      <input
                        type="radio"
                        name="valueSelector"
                        value="TRANFORMER"
                        checked={selectedModel === "TRANFORMER"}
                        onChange={() => handleSelect("TRANFORMER")}
                      />
                      TRANFORMER model
                    </label>
                  </div>
                </div>
                <div>
                  <button
                    className="btnGenerate"
                    onClick={handleGenerateCaption}
                  >
                    Generate Caption
                  </button>
                </div>
              </div>
              <LeftBar onDataFromUser={handleTestImage} />
            </div>
          </div>
        )}

        {bool && <Result img={selectedFile} model={selectedModel} />}
      </div>
    </>
  );
};

export default ImageCaptionGenerator;
