import React from "react";
import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import Content from "./components/Content";
import CTA from "./components/GetStarted";
import GetStarted from "./components/CTA";
import Footer from "./components/Footer";
import Stats from "./components/Stats";
import AnalyzeNow from "./components/AnalyzeNow";
import "./index.css";

const App = () => {
  return (
    <div className="bg-gradient-to-b from-black via-gray-900 to-gray-800 text-white overflow-x-hidden scroll-smooth">
      <Navbar />

      {/* Hero Section */}
      <section id="home" className="min-h-screen flex items-center justify-center relative">
        <Hero />
      </section>

        <Stats/>

      {/* Content Section */}
      <section id="howitworks" className="py-20 px-8 bg-gray-900">
        <Content />
      </section>

      {/* Get Started Section */}
      <section id="ai" className="py-20 px-8 bg-gray-950">
        <GetStarted />
      </section>

      {/* CTA Section */}
      <section id="about" className="py-20 px-8 bg-gradient-to-r from-gray-800 to-black">
        <CTA />
      </section>

      <section id="analyze" className="py-20 px-8 bg-gray-950">
        <AnalyzeNow/>
      </section>

      {/* Footer */}
      <Footer />
    </div>
  );
};

export default App;
