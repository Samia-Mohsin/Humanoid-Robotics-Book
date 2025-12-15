import React, { useContext } from 'react';
import { AuthContext } from '../contexts/AuthContext';

const Navbar = () => {
  const { user, logout } = useContext(AuthContext);

  const handleLogout = async () => {
    await logout();
    window.location.href = '/';
  };

  return (
    <div className="navbar-component">
      {user ? (
        <div className="user-menu">
          <span className="user-email">{user.email}</span>
          <button onClick={handleLogout} className="button button--sm button--secondary">
            Logout
          </button>
          <a href="/profile" className="button button--sm button--outline">
            Profile
          </a>
        </div>
      ) : (
        <div className="auth-menu">
          <a href="/login" className="button button--sm button--secondary">
            Log In
          </a>
          <a href="/signup" className="button button--sm button--primary margin-left--sm">
            Sign Up
          </a>
        </div>
      )}
    </div>
  );
};

export default Navbar;