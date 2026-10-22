import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import Bench from "./Bench";
import "./styles.css";

const isBench = new URLSearchParams(window.location.search).has("bench");

createRoot(document.getElementById("root")!).render(
  <StrictMode>{isBench ? <Bench /> : <App />}</StrictMode>,
);
