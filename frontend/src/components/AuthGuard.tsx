// frontend/src/components/AuthGuard.tsx
import React, { useEffect } from 'react';
import { useUser } from '../contexts/UserContext';
import { useNavigate, useLocation } from 'react-router-dom';
import { toast } from 'react-hot-toast';

interface AuthGuardProps {
  children: React.ReactNode;
  requireAuth?: boolean; // If true, requires authentication; if false, requires being logged out
  fallbackPath?: string; // Where to redirect if auth check fails
}

const AuthGuard: React.FC<AuthGuardProps> = ({
  children,
  requireAuth = true,
  fallbackPath = '/login'
}) => {
  const { user, loading } = useUser();
  const navigate = useNavigate();
  const location = useLocation();
  const isAuthenticated = !!user && !loading;

  useEffect(() => {
    // If we require auth but user is not authenticated, redirect
    if (requireAuth && !isAuthenticated && !loading) {
      toast.error('Please log in to access this feature');
      navigate(fallbackPath, {
        replace: true,
        state: { from: location.pathname }
      });
    }

    // If we require being logged out but user is authenticated, redirect
    if (!requireAuth && isAuthenticated && !loading) {
      navigate('/', { replace: true });
    }
  }, [isAuthenticated, loading, requireAuth, navigate, location, fallbackPath]);

  // Show loading state while checking auth status
  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  // If auth check fails (user not authenticated when auth required), don't render children
  if (requireAuth && !isAuthenticated) {
    return null;
  }

  // If user shouldn't be authenticated but is, don't render children
  if (!requireAuth && isAuthenticated) {
    return null;
  }

  // Otherwise, render the protected content
  return <>{children}</>;
};

export default AuthGuard;