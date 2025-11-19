import React, { useState } from "react";
import { Close, Logo, Menu } from "../assets";
import { navLinks } from "../constants";

const Navbar = () => {
  const [toggle, setToggle] = useState(false);

  const handleScroll = (id) => {
    const section = document.getElementById(id);
    if (section) {
      section.scrollIntoView({ behavior: "smooth" });
      setToggle(false); // Close mobile menu after click
    }
  };

  return (
    <nav className="w-full flex py-6 justify-between items-center px-10 top-0 left-0 z-20 bg-transparent fixed">
      <img src={Logo} alt="MindAI" className="w-[124px] h-[124px]" />

      {/* Desktop Menu */}
      <ul className="list-none sm:flex hidden justify-end items-center flex-1">
        {navLinks.map((nav) => (
          <li
            key={nav.id}
            className="font-medium cursor-pointer text-white hover:text-cyan-400 mx-4"
            onClick={() => handleScroll(nav.id)}
          >
            {nav.title}
          </li>
        ))}
      </ul>

      {/* Mobile Menu */}
      <div className="sm:hidden flex flex-1 justify-end items-center">
        <img
          src={toggle ? Close : Menu}
          alt="menu"
          className="w-[36px] h-[36px] object-contain"
          onClick={() => setToggle((prev) => !prev)}
        />

        <div
          className={`${
            toggle ? "flex" : "hidden"
          } p-6 bg-black/80 absolute top-20 right-0 mx-4 my-2 min-w-[160px] rounded-xl sidebar`}
        >
          <ul className="list-none flex flex-col justify-end items-center flex-1">
            {navLinks.map((nav) => (
              <li
                key={nav.id}
                className="font-medium cursor-pointer text-white hover:text-cyan-400 mb-4"
                onClick={() => handleScroll(nav.id)}
              >
                {nav.title}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
