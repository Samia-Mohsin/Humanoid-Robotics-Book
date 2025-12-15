import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';

const SigninPage = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });

  // Safely get login function from context, defaulting to a mock function if context is not available
  let login;
  try {
    const auth = useAuth();
    login = auth?.login || (() => {});
  } catch {
    login = () => {};
  }

  // For Docusaurus, we'll use a mock navigation during build, real navigation during runtime
  const navigate = (path) => {
    if (typeof window !== 'undefined') {
      // In browser environment, we can use navigation
      window.location.hash = path;
    }
    // During build, navigation is a no-op
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // In a real implementation, this would call your backend API
    // For now, we'll simulate the signin process
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Mock login
      login({
        id: 'user_' + Date.now(),
        email: formData.email,
        name: 'Test User'
      }, 'mock_token_' + Date.now());

      navigate('/dashboard');
    } catch (error) {
      console.error('Signin error:', error);
      alert('Signin failed. Please try again.');
    }
  };

  return (
    <div className="auth-form">
      <h2>Sign In</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            required
          />
        </div>

        <button type="submit">Sign In</button>
      </form>

      <p style={{ marginTop: '1rem', textAlign: 'center' }}>
        Don't have an account? <a href="/auth/signup">Sign up</a>
      </p>
    </div>
  );
};

export default SigninPage;