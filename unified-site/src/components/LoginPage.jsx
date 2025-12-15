import React, { useState, useContext } from 'react';
import { AuthContext } from '../contexts/AuthContext';
import { useNavigate, useLocation } from 'react-router-dom';

const LoginPage = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [errors, setErrors] = useState({});
  const { login } = useContext(AuthContext);
  const navigate = useNavigate();
  const location = useLocation();

  const from = location.state?.from?.pathname || '/';

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.email) newErrors.email = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(formData.email)) newErrors.email = 'Email is invalid';

    if (!formData.password) newErrors.password = 'Password is required';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    try {
      const result = await login(formData.email, formData.password);

      if (result.success) {
        navigate(from, { replace: true }); // Redirect to intended page or home
      } else {
        setErrors({ general: result.message || 'Login failed' });
      }
    } catch (error) {
      setErrors({ general: error.message || 'An error occurred during login' });
    }
  };

  return (
    <div className="login-page">
      <div className="container">
        <div className="row justify-content-center">
          <div className="col-md-6">
            <h2>Log In to Physical AI & Humanoid Robotics</h2>
            <p className="text-muted">Access your personalized learning experience</p>

            {errors.general && (
              <div className="alert alert-danger">{errors.general}</div>
            )}

            <form onSubmit={handleSubmit}>
              <div className="mb-3">
                <label htmlFor="email" className="form-label">Email</label>
                <input
                  type="email"
                  className={`form-control ${errors.email ? 'is-invalid' : ''}`}
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                />
                {errors.email && <div className="invalid-feedback">{errors.email}</div>}
              </div>

              <div className="mb-3">
                <label htmlFor="password" className="form-label">Password</label>
                <input
                  type="password"
                  className={`form-control ${errors.password ? 'is-invalid' : ''}`}
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleInputChange}
                />
                {errors.password && <div className="invalid-feedback">{errors.password}</div>}
              </div>

              <button type="submit" className="btn btn-primary w-100">
                Log In
              </button>
            </form>

            <div className="text-center mt-3">
              <p>
                Don't have an account? <a href="/signup">Sign up</a>
              </p>
              <p>
                <a href="/forgot-password">Forgot password?</a>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;