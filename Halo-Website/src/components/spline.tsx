// components/SplineScene.tsx
"use client"; // ✅ must be client

import React from "react";
import Spline from "@splinetool/react-spline/next";

const SplineScene: React.FC = () => {
  return (
    <Spline scene="https://prod.spline.design/aEiUQHZcIWEDlaxZ/scene.splinecode" />
  );
};

export default SplineScene;
