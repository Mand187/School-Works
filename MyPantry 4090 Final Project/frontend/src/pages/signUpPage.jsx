import { useState } from "react";
import { Link } from "react-router-dom";
import axios from 'axios'

function Signup() {
    const [name,     setname]        = useState()
    const [email,    setemail]       = useState()
    const [password, setPassword]    = useState()

    const handleSubmit = (e) => {
        e.preventDefault()
        axios.post('http://localhost:3000/register', (name, email, password))
        .then(result => console.log(result))
        .catch(err=> console.log(err))
    }

    return (
        <div className="d-flex justify-content-center align-items-center bg-secondary vh-100">
            <div className="bg-white p-3 rounded col-10 col-md-6 col-lg-4">
                <h2>Register</h2>
                <form on onSubmit={handleSubmit}>
                    <div className="mb-3">
                        <label htmlFor="name">
                            <strong>Name</strong>
                        </label>
                        <input
                            type="text"
                            placeholder="Enter Name"
                            autoComplete="off"
                            name="name"
                            id="name"
                            className="form-control rounded-0"
                            onChange={(e) => setname(e.target.value)}
                        />
                    </div>
                    <div className="mb-3">
                        <label htmlFor="email">
                            <strong>Email</strong>
                        </label>
                        <input
                            type="email"
                            placeholder="Enter a Valid Email"
                            autoComplete="off"
                            name="email"
                            id="email"
                            className="form-control rounded-0"
                            onChange={(e) => setemail(e.target.value)}
                        />
                    </div>
                    <div className="mb-3">
                        <label htmlFor="password">
                            <strong>Password</strong>
                        </label>
                        <input
                            type="password"
                            placeholder="Enter a Password"
                            autoComplete="off"
                            name="password"
                            id="password"
                            className="form-control rounded-0"
                            onChange={(e) => setPassword(e.target.value)}
                        />
                    </div>
                    <button type="submit" className="btn btn-success w-100 rounded-0">
                        Register
                    </button>
                </form>
                <p className="mt-3">Already Have an Account?</p>
                <Link to="/login" className="btn btn-default border w-100 bg-light rounded-0 text-decoration-none">
                    Login
                </Link>
            </div>
        </div>
    );
}

export default Signup;
