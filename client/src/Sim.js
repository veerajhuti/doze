import React, { useState, useEffect, useRef } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import Button from '@mui/joy/Button';
import "react-toastify/dist/ReactToastify.css";

function Sim({ onExit }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = canvas.getContext("webgl");
    
    const camera = { x: 0, y: 0, z: 2, yaw: 0 }; // yaw -> rotation
    function getForwardVector(yaw) { // rotation -> direction
      return {
        x: Math.sin(yaw), // sideways
        z: Math.cos(yaw) // fwd
      };
    }
    
    if (!gl) {
      alert("WebGL not supported");
      return;
    }

    // https://developer.mozilla.org/en-US/docs/Web/API/Pointer_Lock_API
    let isPointerLocked = false;

    canvas.addEventListener("click", () => {
      canvas.requestPointerLock();
    });

    document.addEventListener("pointerlockchange", () => { // update when pointer state changes
      isPointerLocked = document.pointerLockElement === canvas;
    });

    canvas.requestPointerLock = canvas.requestPointerLock || canvas.mozRequestPointerLock;
    canvas.onclick = () => {
      canvas.requestPointerLock();
    };

    document.addEventListener("mousemove", (e) => {
    if (!isPointerLocked) {
      return;
    }
  });

    // glsl -> vertex shader
    const vsSource = `
    attribute vec4 aVertexPosition;

    uniform mat4 uProjectionMatrix;
    uniform mat4 uViewMatrix;

    void main() {
      gl_Position = uProjectionMatrix * uViewMatrix * aVertexPosition;
    }
    `;
    // glsl -> fragment shader -> set triangle to white
    const fsSource = `
      void main() {
        gl_FragColor = vec4(1, 1, 1, 1);
      }
    `;

    const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
    const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);
    
    const shaderProgram = gl.createProgram();

    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);
    gl.useProgram(shaderProgram);

    // const cameraLocation = gl.getUniformLocation(shaderProgram, "uCamera"); // we want the camera to move not the triangle, with the world still
    const projectionLocation = gl.getUniformLocation(shaderProgram, "uProjectionMatrix");
    const viewLocation = gl.getUniformLocation(shaderProgram, "uViewMatrix");
    
    const aspect = canvas.width / canvas.height;
    // https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_model_view_projection
    const projectionMatrix = createPerspectiveMatrix((60 * Math.PI) / 180, aspect, 0.1, 100); // 3d -> 2d

    gl.uniformMatrix4fv(
      projectionLocation,
      false,
      projectionMatrix
    );

    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Float32Array
    // triangle (static) to turn into cube/car
    const vertices = new Float32Array([
      0,  1, -5,
    -1, -1, -5,
      1, -1, -5
    ]);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer); // active buffer
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    
    // send js attribute to set vertex data location in shader
    const positionLocation = gl.getAttribLocation(
      shaderProgram,
      "aVertexPosition"
    );

    gl.enableVertexAttribArray(positionLocation);

    // each vertex is 3 floats (x,y,z), not normalized, 0 stride (space btwn -> tightly packed)
    gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);

    gl.clearColor(0.05, 0.06, 0.10, 1); // set clear colour
    gl.clear(gl.COLOR_BUFFER_BIT); // clear screen

    // gl.drawArrays(gl.TRIANGLES, 0, 3); // static
    function render() {
      const speed = 0.05;
      const viewMatrix = createViewMatrix(camera);

      const forward = getForwardVector(camera.yaw);

      if (keys["w"]) {
        camera.x += forward.x * speed;
        camera.z += forward.z * speed;
      }
      if (keys["s"]) {
        camera.x -= forward.x * speed;
        camera.z -= forward.z * speed;
      }
      if (keys["a"]) {
        camera.x += forward.z * speed;
        camera.z -= forward.x * speed;
      }
      if (keys["d"]) {
        camera.x -= forward.z * speed;
        camera.z += forward.x * speed;
      }

      gl.uniformMatrix4fv(projectionLocation, false, projectionMatrix);
      gl.uniformMatrix4fv(viewLocation, false, viewMatrix);

      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      requestAnimationFrame(render);
    }

    const keys = {};

    window.addEventListener("keydown", (e) => {
      keys[e.key] = true;
    });

    window.addEventListener("keyup", (e) => {
      keys[e.key] = false;
    });

    render();
  }, []);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        overflow: "hidden",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        background: "#0b0c14",
      }}
    >
      {/* blurred background */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "rgba(10, 10, 20, 0.6)",
          backdropFilter: "blur(12px)",
        }}
      />

      {/* back button */}
      <div
        style={{
          position: "absolute",
          top: 16,
          left: 16,
          zIndex: 10,
        }}
      >
        <Button
          onClick={onExit}
          style={{
            background: "linear-gradient(to bottom, #8496FF, #8873FF)",
            border: "none",
            color: "white",
            padding: "8px 12px",
            borderRadius: 8,
          }}
        >
          ← Back
        </Button>
      </div>

      <div
        style={{
          position: "absolute",
          top: 16,
          right: 16,
          zIndex: 10,
          width: 180,
          padding: "10px 12px",
          borderRadius: "12px",
          background: "",
          border: "1px solid rgba(255,255,255,0.08)",
          backdropFilter: "blur(10px)",
          color: "rgba(255,255,255,0.85)",
          fontFamily: '"Baloo", sans-serif',
          fontSize: "13px",
          lineHeight: 1.4,
        }}
      >
        *Detector webcam here*
      </div>

      {/* sim window */}
      <div
        style={{
          width: "min(900px, 90vw)",
          height: "min(600px, 70vh)",
          borderRadius: "16px",
          overflow: "hidden",
          background: "linear-gradient(to bottom, #8496FF, #8873FF)",
          border: "1px solid rgba(255,255,255,0.08)",
          boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
          zIndex: 5,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: "100%",
            height: "100%",
            display: "block",
          }}
        />
      </div>
    </div>
  );
}

// create empty shader
function loadShader(gl, type, source) {
  const shader = gl.createShader(type);

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  return shader;
}

function createPerspectiveMatrix(fov, aspect, near, far) {
  const f = 1.0 / Math.tan(fov / 2);

  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0
  ]);
}

function createViewMatrix(camera) {
  const cosY = Math.cos(camera.yaw);
  const sinY = Math.sin(camera.yaw);

  const x = camera.x;
  const y = camera.y;
  const z = camera.z;

  return new Float32Array([
    cosY, 0, -sinY, 0,
    0,    1, 0,     0,
    sinY, 0, cosY,  0,

    -(cosY * x - sinY * z),
    -y,
    -(sinY * x + cosY * z),
    1
  ]);
}
export default Sim;