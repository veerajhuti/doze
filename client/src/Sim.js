import React, { useState, useEffect, useRef } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import Button from '@mui/joy/Button';
import "react-toastify/dist/ReactToastify.css";

function draw() {
  orbitControl()
}

function Sim({ onExit }) {
  const canvasRef = useRef(null);
  useEffect(() => {
      const canvas = canvasRef.current;
      // Initialize the GL context
      const gl = canvas.getContext("webgl");

      // Only continue if WebGL is available and working
      if (gl === null) {
        alert(
          "Unable to initialize WebGL. Your browser or machine may not support it.",
        );
        return;
      }
    // Set clear color to black, fully opaque
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    // Clear the color buffer with specified clear color
    gl.clear(gl.COLOR_BUFFER_BIT);
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

export default Sim;