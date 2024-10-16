import React, { useState } from "react";
import "./Header.scss";


import { RiMenu3Line } from "react-icons/ri";
import { MdClose } from "react-icons/md";
import { Link, useNavigate } from "react-router-dom";
function Header() {
  const [show, setShow] = useState(false);
  const [sticky, setSticky] = useState(false);

  const navigate = useNavigate();
  const handleHome = () => {
    navigate("/");
  };

  const handleScroll = () => {
    if (window.scrollY > 150) {
      setSticky(true);
    } else {
      setSticky(false);
    }
  };

  window.addEventListener("scroll", handleScroll);
  return (
    <div className={sticky ? "header active" : "header"}>
      <div className="left">
        <Link to={"/"} className="link">
          <h1>COINLENS</h1>
        </Link>
      </div>
      <div className="middle">
        <ul>
          <li>
            <a href="#home" onClick={handleHome}>
              Home
            </a>
          </li>
          <li>
            <a href="#home">Market</a>
          </li>
          <li>
            <a href="#home">Choose Us</a>
          </li>
          <li>
            <a href="#home">Join</a>
          </li>
        </ul>
      </div>
      <h3 onClick={() => setShow(!show)}>
        <RiMenu3Line />
      </h3>
      {show && (
        <div className="mobile">
          <div className="ul">
            <ul>
              <li>
                <a
                  href="#home"
                  onClick={() => {
                    setShow(false);
                    navigate("/");
                  }}
                >
                  Home
                </a>
              </li>
              <li>
                <a href="#home" onClick={() => setShow(false)}>
                  Market
                </a>
              </li>
              <li>
                <a href="#home" onClick={() => setShow(false)}>
                  Choose Us
                </a>
              </li>
              <li>
                <a href="#home" onClick={() => setShow(false)}>
                  Join
                </a>
              </li>
            </ul>
          </div>
          <h4 onClick={() => setShow(false)}>
            <MdClose />
          </h4>
        </div>
      )}
    </div>
  );
}

export default Header;
