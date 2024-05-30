import { useState } from "react";

function Signup() {
    return (
        <div className="d-flex justify-content-center align-items-center bg-secondary vh-100">
            <div className="bg-white p-3 rounded col-10 col-md-6 col-lg-4">
                <h2>Register</h2>
                <form>
                    <div className="mb-3">
                        <label htmlFor="name">
                            <strong>Name</strong>
                        </label>
                        <input
                            type="text"
                            placeholder="Enter Name"
                            autoComplete="off"
                            name="name"
                            className="form-control rounded-0"
                        />
                    </div>
                    <div className="mb-3">
                        <label htmlFor="Email">
                            <strong>Email</strong>
                        </label>
                        <input
                            type="email"
                            placeholder="Enter a Valid Email"
                            autoComplete="off"
                            name="Email"
                            className="form-control rounded-0"
                        />
                    </div>
                    <div className="mb-3">
                        <label htmlFor="Password">
                            <strong>Password</strong>
                        </label>
                        <input
                            type="password"
                            placeholder="Enter a Password"
                            autoComplete="off"
                            name="password"
                            className="form-control rounded-0"
                        />
                    </div>
                    <button type="submit" className="btn btn-success w-100 rounded-0">
                        Register
                    </button>
                    <p className="mt-3">Already Have an Account?</p>
                    <button className="btn btn-default border w-100 bg-light rounded-0 text-decoration-none">
                        Login
                    </button>
                </form>
            </div>
        </div>
    );
}

export default Signup;
