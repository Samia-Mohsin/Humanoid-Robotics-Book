import React, { useState, useContext } from 'react';
import { AuthContext } from '../contexts/AuthContext';
import { PersonalizationContext } from '../contexts/PersonalizationContext';
import { useNavigate } from 'react-router-dom';

const SignupPage = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    full_name: '',
    // Background questions
    programming_languages: [],
    ai_ml_experience: '',
    hardware_experience: '',
    learning_goals: '',
    gpu_access: false,
    robotics_kit_experience: '',
    preferred_topics: []
  });
  const [errors, setErrors] = useState({});
  const { signup } = useContext(AuthContext);
  const { updateUserPreferences } = useContext(PersonalizationContext);
  const navigate = useNavigate();

  const programmingOptions = ['Python', 'C++', 'ROS', 'JavaScript', 'Java', 'Other'];
  const topicOptions = ['AI', 'Simulation', 'Control Systems', 'Hardware', 'Robotics', 'Other'];

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleArrayChange = (e, field) => {
    const { value, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [field]: checked
        ? [...prev[field], value]
        : prev[field].filter(item => item !== value)
    }));
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.email) newErrors.email = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(formData.email)) newErrors.email = 'Email is invalid';

    if (!formData.password) newErrors.password = 'Password is required';
    else if (formData.password.length < 6) newErrors.password = 'Password must be at least 6 characters';

    if (!formData.full_name) newErrors.full_name = 'Full name is required';

    if (!formData.ai_ml_experience) newErrors.ai_ml_experience = 'Please select your AI/ML experience level';
    if (!formData.hardware_experience) newErrors.hardware_experience = 'Please select your hardware experience level';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    try {
      // Sign up the user
      const result = await signup(formData.email, formData.password, formData.full_name);

      if (result.success) {
        // Update user preferences with background information
        await updateUserPreferences({
          programming_languages: formData.programming_languages,
          ai_ml_experience: formData.ai_ml_experience,
          hardware_experience: formData.hardware_experience,
          learning_goals: formData.learning_goals,
          gpu_access: formData.gpu_access,
          robotics_kit_experience: formData.robotics_kit_experience,
          preferred_topics: formData.preferred_topics
        });

        navigate('/dashboard'); // Redirect to dashboard or home
      } else {
        setErrors({ general: result.message || 'Signup failed' });
      }
    } catch (error) {
      setErrors({ general: error.message || 'An error occurred during signup' });
    }
  };

  return (
    <div className="signup-page">
      <div className="container">
        <div className="row justify-content-center">
          <div className="col-md-6">
            <h2>Sign Up for Physical AI & Humanoid Robotics</h2>
            <p className="text-muted">Join our educational platform and get personalized content</p>

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
                <label htmlFor="full_name" className="form-label">Full Name</label>
                <input
                  type="text"
                  className={`form-control ${errors.full_name ? 'is-invalid' : ''}`}
                  id="full_name"
                  name="full_name"
                  value={formData.full_name}
                  onChange={handleInputChange}
                />
                {errors.full_name && <div className="invalid-feedback">{errors.full_name}</div>}
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

              <hr className="my-4" />

              <h4>Your Background</h4>
              <p className="text-muted">Help us personalize your learning experience</p>

              <div className="mb-3">
                <label className="form-label">Programming Languages Known</label>
                {programmingOptions.map(lang => (
                  <div className="form-check" key={lang}>
                    <input
                      className="form-check-input"
                      type="checkbox"
                      id={`lang-${lang}`}
                      value={lang}
                      checked={formData.programming_languages.includes(lang)}
                      onChange={(e) => handleArrayChange(e, 'programming_languages')}
                    />
                    <label className="form-check-label" htmlFor={`lang-${lang}`}>
                      {lang}
                    </label>
                  </div>
                ))}
              </div>

              <div className="mb-3">
                <label htmlFor="ai_ml_experience" className="form-label">AI/ML Experience Level</label>
                <select
                  className={`form-select ${errors.ai_ml_experience ? 'is-invalid' : ''}`}
                  id="ai_ml_experience"
                  name="ai_ml_experience"
                  value={formData.ai_ml_experience}
                  onChange={handleInputChange}
                >
                  <option value="">Select your experience level</option>
                  <option value="none">No experience</option>
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                </select>
                {errors.ai_ml_experience && <div className="invalid-feedback">{errors.ai_ml_experience}</div>}
              </div>

              <div className="mb-3">
                <label htmlFor="hardware_experience" className="form-label">Hardware Experience Level</label>
                <select
                  className={`form-select ${errors.hardware_experience ? 'is-invalid' : ''}`}
                  id="hardware_experience"
                  name="hardware_experience"
                  value={formData.hardware_experience}
                  onChange={handleInputChange}
                >
                  <option value="">Select your experience level</option>
                  <option value="none">No experience</option>
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                </select>
                {errors.hardware_experience && <div className="invalid-feedback">{errors.hardware_experience}</div>}
              </div>

              <div className="mb-3">
                <label htmlFor="robotics_kit_experience" className="form-label">Robotics Kit Experience</label>
                <select
                  className="form-select"
                  id="robotics_kit_experience"
                  name="robotics_kit_experience"
                  value={formData.robotics_kit_experience}
                  onChange={handleInputChange}
                >
                  <option value="">Select your experience</option>
                  <option value="none">None</option>
                  <option value="basic">Basic (e.g., Arduino, Raspberry Pi)</option>
                  <option value="intermediate">Intermediate (e.g., ROS robots)</option>
                  <option value="advanced">Advanced (e.g., custom robots)</option>
                </select>
              </div>

              <div className="mb-3">
                <label htmlFor="learning_goals" className="form-label">Learning Goals</label>
                <textarea
                  className="form-control"
                  id="learning_goals"
                  name="learning_goals"
                  rows="3"
                  value={formData.learning_goals}
                  onChange={handleInputChange}
                  placeholder="What do you hope to achieve with this course?"
                ></textarea>
              </div>

              <div className="mb-3 form-check">
                <input
                  type="checkbox"
                  className="form-check-input"
                  id="gpu_access"
                  name="gpu_access"
                  checked={formData.gpu_access}
                  onChange={handleInputChange}
                />
                <label className="form-check-label" htmlFor="gpu_access">
                  I have access to a GPU for AI training
                </label>
              </div>

              <div className="mb-3">
                <label className="form-label">Preferred Topics</label>
                {topicOptions.map(topic => (
                  <div className="form-check" key={topic}>
                    <input
                      className="form-check-input"
                      type="checkbox"
                      id={`topic-${topic}`}
                      value={topic}
                      checked={formData.preferred_topics.includes(topic)}
                      onChange={(e) => handleArrayChange(e, 'preferred_topics')}
                    />
                    <label className="form-check-label" htmlFor={`topic-${topic}`}>
                      {topic}
                    </label>
                  </div>
                ))}
              </div>

              <button type="submit" className="btn btn-primary w-100">
                Sign Up
              </button>
            </form>

            <div className="text-center mt-3">
              <p>
                Already have an account? <a href="/login">Log in</a>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;