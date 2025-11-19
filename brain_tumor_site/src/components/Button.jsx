import React, { useState } from "react";
import UploadCard from "./UploadCard";

const Button = ({ styles }) => {
  const [showModal, setShowModal] = useState(false);

  const handleClick = () => {
    setShowModal(true);
  };

  const handleClose = () => {
    setShowModal(false);
  };

  return (
    <>
      <button
        onClick={handleClick}
        className={`font-poppins bg-cyan-400 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition ${styles}`}
      >
        Analyze
      </button>

      {showModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-gray-900 p-6 rounded-2xl shadow-2xl relative max-w-lg w-full">
            <button
              onClick={handleClose}
              className="absolute top-3 right-4 text-gray-400 hover:text-red-500 text-2xl font-bold"
            >
              ✖
            </button>
            <UploadCard />
          </div>
        </div>
      )}
    </>
  );
};

export default Button;
