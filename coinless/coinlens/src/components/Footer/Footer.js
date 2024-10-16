import React from "react";
import "./Footer.scss";
import {  bsLinkedin, bsDiscord, bsGithub, BsTwitterX } from "react-icons/bs";
import {bsX} from "react-icons/bs";
function Footer() {
  return (
    <div className="footer">
      <div className="content">
        <a
          href="https://x.com/HarshaReddyRow1"
          target="_blank"
          rel="noreferrer"
        >
          <BsTwitterX />
        </a>
        <a
          href="https://www.linkedin.com/in/harsha-vardhan-reddy-3a3169247/"
          target="_blank"
          rel="noreferrer"
        >
          <bsLinkedin/>
        </a>
        <a href="#home">
          <bsDiscord />
        </a>
        <a
          href="https://github.com/harshareddi-7"
          target="_blank"
          rel="noreferrer"
        >
          <bsGithub />
        </a>
      </div>
      <div className="page">
        <p>Privacy</p>
        <p>Terms of Use</p>
      </div>
    </div>
  );
}

export default Footer;
