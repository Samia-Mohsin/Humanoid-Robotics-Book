// frontend/src/components/ui/toast.tsx
import React from 'react';

interface ToastProps {
  message: string;
  type?: 'default' | 'success' | 'error' | 'warning' | 'info';
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

const Toast: React.FC<ToastProps> = ({
  message,
  type = 'default',
  open = true,
  onOpenChange
}) => {
  if (!open) return null;

  const typeStyles = {
    default: 'bg-white border-gray-200 text-gray-800',
    success: 'bg-green-100 border-green-200 text-green-800',
    error: 'bg-red-100 border-red-200 text-red-800',
    warning: 'bg-yellow-100 border-yellow-200 text-yellow-800',
    info: 'bg-blue-100 border-blue-200 text-blue-800'
  };

  return (
    <div
      className={`fixed top-4 right-4 z-50 p-4 rounded-md shadow-lg border ${typeStyles[type]} transition-all duration-300`}
    >
      <div className="flex items-start">
        <p className="text-sm font-medium">{message}</p>
        <button
          onClick={() => onOpenChange?.(false)}
          className="ml-4 text-gray-500 hover:text-gray-700"
        >
          ×
        </button>
      </div>
    </div>
  );
};

export { Toast };