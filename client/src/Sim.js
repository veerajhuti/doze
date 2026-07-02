import React, { useState, useEffect, useRef } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import Button from '@mui/joy/Button';
import "react-toastify/dist/ReactToastify.css";

function Sim({ onExit }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const camera = { x: 0, y: 0, z: 2 };

    let offsetX = 0; // we want WASD to change offset X,Y,Z

    const canvas = canvasRef.current;
    const gl = canvas.getContext("webgl");

    if (!gl) {
      alert("WebGL not supported");
      return;
    }

    // glsl -> vertex shader
    const vsSource = `
      attribute vec4 aVertexPosition;

      uniform vec3 uCamera;

      void main() {
        vec4 position = aVertexPosition;

        position.x = position.x - uCamera.x;
        position.y = position.y - uCamera.y;
        position.z = position.z - uCamera.z;

        gl_Position = position;
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

    const cameraLocation = gl.getUniformLocation(shaderProgram, "uCamera"); // we want the camera to move not the triangle, with the world still

    // triangle (static)
    const vertices = new Float32Array([
      0.0,  1.0,  0.0,
    -1.0, -1.0,  0.0,
      1.0, -1.0,  0.0,
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

    gl.clearColor(0, 0, 0, 1); // set clear colour
    gl.clear(gl.COLOR_BUFFER_BIT); // clear screen

    // gl.drawArrays(gl.TRIANGLES, 0, 3); // static
    function render() {
      const speed = 0.05;
      if (keys["w"]) camera.z -= speed;
      if (keys["s"]) camera.z += speed;
      if (keys["a"]) camera.x -= speed;
      if (keys["d"]) camera.x += speed;
      
      gl.uniform3f(cameraLocation, camera.x, camera.y, camera.z);

      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);

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
    <>
      <Button onClick={onExit}>
        ← Back to Doze
      </Button>

      <canvas
        ref={canvasRef}
        width={640}
        height={480}
      />
    </>
  );
}

// create empty shader
function loadShader(gl, type, source) {
  const shader = gl.createShader(type);

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  return shader;
}

export default Sim;