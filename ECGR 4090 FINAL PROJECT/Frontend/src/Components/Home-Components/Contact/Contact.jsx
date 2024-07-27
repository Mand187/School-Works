import React from 'react';
import './Contact.css';

const Contact = () => {
  return (
    <div className="contactForm">
      <div className="contact-col">
        <h3>Send us a message</h3>
        <p>
          Feel free to reach out through the contact form or find our contact information below.
          Your feedback, questions, and suggestions are important to us as we strive to provide
          exceptional service to our website users.
        </p>
        <ul>
          <li>Daniel Myers | <a href="mailto:cmyers54@uncc.edu">cmyers54@uncc.edu</a></li>
          <li>Matthew Anderson | <a href="mailto:mande137@uncc.edu">mande137@uncc.edu</a></li>
          <li>Hunter Burnett | <a href="mailto:hburnet7@charlotte.edu">hburnet7@charlotte.edu</a></li>
          <li>Aidan Cowan | <a href="mailto:acowan8@uncc.edu">acowan8@uncc.edu</a></li>
        </ul>
      </div>
      <div className='contact-col'>
        <form>
          <label>Your Name</label>
          <input type="text" name='name' placeholder='Enter your name' required />
          <label>Phone Number</label>
          <input type='tel' name='phone' placeholder='Enter your mobile phone number' />
          <label>Write your messages here</label>
          <textarea name='message' rows={6} placeholder='Enter your message' required></textarea>
          <button type='submit' className='Submit-btn'>Submit</button>
        </form>
      </div>
    </div>
  );
}

export default Contact;
