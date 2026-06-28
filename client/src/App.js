import Vanilla from "./Vanilla";
import Sim from "./Sim";
import React, { useState, useEffect, useRef } from 'react';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import { ToastContainer, toast } from 'react-toastify';
import Button from '@mui/joy/Button';
import "react-toastify/dist/ReactToastify.css";

function App() {
  const [mode, setMode] = useState("vanilla");

  if (mode === "sim") {
    return <Sim onExit={() => setMode("vanilla")} />;
  }

  return (
    <Vanilla onStartSim={() => setMode("sim")} />
  );
}

export default App;