/*
═══════════════════════════════════════════════════════════════
  Civic Issue Urgency Classifier - Main JavaScript
  iOS 26 Liquid Design Interactions
═══════════════════════════════════════════════════════════════
*/

// Global state
const state = {
  isClassifying: false,
  currentResult: null,
};

// ============================================
//  DOM READY
// ============================================
document.addEventListener('DOMContentLoaded', () => {
  initializeApp();
  initializeAnimations();
  initializeFormHandlers();
  initializeScrollEffects();
});

// ============================================
//  INITIALIZATION
// ============================================
function initializeApp() {
  console.log('🏛️ Civic Issue Urgency Classifier initialized');
  
  // Add navbar scroll effect
  const navbar = document.querySelector('.navbar');
  if (navbar) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
      } else {
        navbar.classList.remove('scrolled');
      }
    });
  }
  
  // Initialize character counter
  const textarea = document.getElementById('issueDescription');
  if (textarea) {
    textarea.addEventListener('input', updateCharCount);
    updateCharCount.call(textarea);
  }
}

// ============================================
//  FORM HANDLING
// ============================================
function initializeFormHandlers() {
  const form = document.getElementById('classifyForm');
  if (form) {
    form.addEventListener('submit', handleFormSubmit);
  }
  
  // Add real-time validation
  const inputs = document.querySelectorAll('.form-input, .form-textarea');
  inputs.forEach(input => {
    input.addEventListener('blur', validateField);
    input.addEventListener('focus', clearFieldError);
  });
}

async function handleFormSubmit(e) {
  e.preventDefault();
  
  if (state.isClassifying) return;
  
  const form = e.target;
  const formData = new FormData(form);
  
  // Validate form
  if (!validateForm(form)) {
    showError('Please fill in all required fields');
    return;
  }
  
  // Show loading state
  showLoading();
  state.isClassifying = true;
  
  try {
    // Prepare data
    const data = {
      text_description: formData.get('description'),
      location_address: formData.get('location') || 'Unknown Location',
      category: formData.get('category') || 'General'
    };
    
    console.log('Sending classification request:', data);
    
    // Send request
    const response = await fetch('/classify-urgency', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data)
    });
    
    console.log('Response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Response error:', errorText);
      throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
    }
    
    const result = await response.json();
    console.log('Classification result:', result);
    state.currentResult = result;
    
    // Show result with animation
    setTimeout(() => {
      hideLoading();
      displayResult(result);
    }, 800);
    
  } catch (error) {
    console.error('Classification error:', error);
    hideLoading();
    showError(`Failed to classify issue: ${error.message || 'Please try again.'}`);
  } finally {
    state.isClassifying = false;
  }
}

// ============================================
//  FORM VALIDATION
// ============================================
function validateForm(form) {
  const description = form.querySelector('[name="description"]');
  
  if (!description.value.trim()) {
    showFieldError(description, 'Please enter a description');
    return false;
  }
  
  if (description.value.trim().length < 10) {
    showFieldError(description, 'Description must be at least 10 characters');
    return false;
  }
  
  return true;
}

function validateField(e) {
  const field = e.target;
  
  if (field.hasAttribute('required') && !field.value.trim()) {
    showFieldError(field, 'This field is required');
  }
}

function showFieldError(field, message) {
  field.classList.add('error');
  field.classList.add('shake');
  
  // Remove existing error message
  const existingError = field.parentElement.querySelector('.error-message');
  if (existingError) {
    existingError.remove();
  }
  
  // Add error message
  const errorDiv = document.createElement('div');
  errorDiv.className = 'error-message';
  errorDiv.textContent = message;
  errorDiv.style.cssText = 'color: #ff3b30; font-size: 0.9rem; margin-top: 0.5rem;';
  field.parentElement.appendChild(errorDiv);
  
  // Remove shake animation after it completes
  setTimeout(() => field.classList.remove('shake'), 500);
}

function clearFieldError(e) {
  const field = e.target;
  field.classList.remove('error');
  
  const errorMessage = field.parentElement.querySelector('.error-message');
  if (errorMessage) {
    errorMessage.remove();
  }
}

// ============================================
//  CHARACTER COUNTER
// ============================================
function updateCharCount() {
  const charCount = document.getElementById('charCount');
  if (charCount) {
    const length = this.value.length;
    charCount.textContent = `${length} characters`;
    
    // Change color based on length
    if (length < 10) {
      charCount.style.color = '#ff3b30';
    } else if (length < 50) {
      charCount.style.color = '#ff9500';
    } else {
      charCount.style.color = '#34c759';
    }
  }
}

// ============================================
//  LOADING STATE
// ============================================
function showLoading() {
  const resultContainer = document.getElementById('resultContainer');
  if (resultContainer) {
    resultContainer.innerHTML = `
      <div class="loading fade-in">
        <div class="loading-spinner"></div>
        <p style="margin-top: 1rem; color: var(--text-muted);">
          Analyzing your civic issue...
        </p>
        <div class="typing-indicator" style="margin-top: 1rem;">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    `;
    resultContainer.classList.remove('hidden');
    
    // Scroll to result
    setTimeout(() => {
      resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
  }
}

function hideLoading() {
  // Loading will be replaced by result
}

// ============================================
//  RESULT DISPLAY
// ============================================
function displayResult(result) {
  const resultContainer = document.getElementById('resultContainer');
  if (!resultContainer) return;
  
  const urgencyClass = `urgency-${result.urgency_level.toLowerCase()}`;
  const urgencyIcon = {
    'HIGH': '🚨',
    'MEDIUM': '⚠️',
    'LOW': '📝'
  }[result.urgency_level] || '📊';
  
  const scorePercent = (result.urgency_score / 10) * 100;
  const confidencePercent = result.confidence * 100;
  
  resultContainer.innerHTML = `
    <div class="result-card bounce-in">
      <h2 style="margin-bottom: 1.5rem; font-size: 2rem;">
        🎯 Classification Result
      </h2>
      
      <div class="urgency-badge ${urgencyClass}">
        ${urgencyIcon} ${result.urgency_level} PRIORITY
      </div>
      
      <div class="result-details">
        <div class="detail-item">
          <div class="detail-label">Urgency Score</div>
          <div class="detail-value">${result.urgency_score.toFixed(1)}/10</div>
          <div class="progress-bar">
            <div class="progress-fill" style="--progress-width: ${scorePercent}%; width: ${scorePercent}%;"></div>
          </div>
        </div>
        
        <div class="detail-item">
          <div class="detail-label">Confidence</div>
          <div class="detail-value">${confidencePercent.toFixed(0)}%</div>
          <div class="progress-bar">
            <div class="progress-fill" style="--progress-width: ${confidencePercent}%; width: ${confidencePercent}%;"></div>
          </div>
        </div>
        
        <div class="detail-item">
          <div class="detail-label">Department</div>
          <div class="detail-value" style="font-size: 1.2rem;">
            🏢 ${result.recommended_department}
          </div>
        </div>
        
        <div class="detail-item">
          <div class="detail-label">Response Time</div>
          <div class="detail-value" style="font-size: 1.2rem;">
            ⏰ ${result.estimated_response_time}
          </div>
        </div>
      </div>
      
      <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(255, 255, 255, 0.05); border-radius: 16px;">
        <h3 style="margin-bottom: 1rem; font-size: 1.3rem;">💭 AI Analysis</h3>
        <p style="color: var(--text-muted); line-height: 1.8;">
          ${result.reasoning}
        </p>
      </div>
      
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-top: 2rem;">
        <div style="text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 12px;">
          <div style="font-size: 2rem;">📝</div>
          <div style="margin-top: 0.5rem; color: var(--text-muted);">Text Analysis</div>
          <div style="font-size: 1.3rem; font-weight: 700; margin-top: 0.3rem;">
            ${(result.text_contribution * 100).toFixed(0)}%
          </div>
        </div>
        
        <div style="text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 12px;">
          <div style="font-size: 2rem;">🖼️</div>
          <div style="margin-top: 0.5rem; color: var(--text-muted);">Image Analysis</div>
          <div style="font-size: 1.3rem; font-weight: 700; margin-top: 0.3rem;">
            ${(result.image_contribution * 100).toFixed(0)}%
          </div>
        </div>
        
        <div style="text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 12px;">
          <div style="font-size: 2rem;">📍</div>
          <div style="margin-top: 0.5rem; color: var(--text-muted);">Location</div>
          <div style="font-size: 1.1rem; font-weight: 700; margin-top: 0.3rem;">
            ${result.location_context}
          </div>
        </div>
        
        <div style="text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 12px;">
          <div style="font-size: 2rem;">⚠️</div>
          <div style="margin-top: 0.5rem; color: var(--text-muted);">Safety Level</div>
          <div style="font-size: 1.1rem; font-weight: 700; margin-top: 0.3rem;">
            ${result.safety_context}
          </div>
        </div>
      </div>
      
      ${getActionPlan(result)}
      
      <div style="margin-top: 2rem; display: flex; gap: 1rem; flex-wrap: wrap;">
        <button onclick="classifyAnother()" class="btn btn-primary">
          🔄 Classify Another Issue
        </button>
        <button onclick="downloadReport()" class="btn btn-secondary">
          📥 Download Report
        </button>
      </div>
    </div>
  `;
  
  resultContainer.classList.remove('hidden');
  
  // Scroll to result
  setTimeout(() => {
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, 100);
}

function getActionPlan(result) {
  const plans = {
    'HIGH': `
      <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, rgba(255, 59, 48, 0.1), rgba(255, 138, 0, 0.1)); border-radius: 16px; border: 2px solid rgba(255, 59, 48, 0.3);">
        <h3 style="margin-bottom: 1rem; font-size: 1.3rem;">🚨 IMMEDIATE ACTION PLAN</h3>
        <ul style="list-style: none; padding: 0;">
          <li style="margin: 0.8rem 0; padding-left: 1.5rem; position: relative;">
            <span style="position: absolute; left: 0;">✅</span>
            Dispatch emergency crew within 1-2 hours
          </li>
          <li style="margin: 0.8rem 0; padding-left: 1.5rem; position: relative;">
            <span style="position: absolute; left: 0;">✅</span>
            Set up safety barriers and warning signs
          </li>
          <li style="margin: 0.8rem 0; padding-left: 1.5rem; position: relative;">
            <span style="position: absolute; left: 0;">✅</span>
            Notify nearby facilities of potential issues
          </li>
          <li style="margin: 0.8rem 0; padding-left: 1.5rem; position: relative;">
            <span style="position: absolute; left: 0;">✅</span>
            Monitor situation until fully resolved
          </li>
        </ul>
      </div>
    `,
    'MEDIUM': `
      <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, rgba(255, 149, 0, 0.1), rgba(255, 204, 0, 0.1)); border-radius: 16px; border: 2px solid rgba(255, 149, 0, 0.3);">
        <h3 style="margin-bottom: 1rem; font-size: 1.3rem;">⚠️ URGENT SCHEDULING PLAN</h3>
        <ul style="list-style: none; padding: 0;">
          <li style="margin: 0.8rem 0; padding-left: 1.5rem; position: relative;">
            <span style="position: absolute; left: 0;">✅</span>
            Add to priority repair queue
          </li>
          <li style="margin: 0.8rem 0; padding-left: 1.5rem; position: relative;">
            <span style="position: absolute; left: 0;">✅</span>
            Schedule repair crew within 24-48 hours
          </li>
          <li style="margin: 0.8rem 0; padding-left: 1.5rem; position: relative;">
            <span style="position: absolute; left: 0;">✅</span>
            Assess if temporary measures needed
          </li>
          <li style="margin: 0.8rem 0; padding-left: 1.5rem; position: relative;">
            <span style="position: absolute; left: 0;">✅</span>
            Update citizen on repair timeline
          </li>
        </ul>
      </div>
    `,
    'LOW': `
      <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, rgba(52, 199, 89, 0.1), rgba(0, 212, 170, 0.1)); border-radius: 16px; border: 2px solid rgba(52, 199, 89, 0.3);">
        <h3 style="margin-bottom: 1rem; font-size: 1.3rem;">📝 ROUTINE MAINTENANCE PLAN</h3>
        <ul style="list-style: none; padding: 0;">
          <li style="margin: 0.8rem 0; padding-left: 1.5rem; position: relative;">
            <span style="position: absolute; left: 0;">✅</span>
            Add to standard maintenance schedule
          </li>
          <li style="margin: 0.8rem 0; padding-left: 1.5rem; position: relative;">
            <span style="position: absolute; left: 0;">✅</span>
            Plan repair within 1-2 weeks
          </li>
          <li style="margin: 0.8rem 0; padding-left: 1.5rem; position: relative;">
            <span style="position: absolute; left: 0;">✅</span>
            Monitor for any changes in condition
          </li>
        </ul>
      </div>
    `
  };
  
  return plans[result.urgency_level] || '';
}

// ============================================
//  UTILITY FUNCTIONS
// ============================================
function classifyAnother() {
  const form = document.getElementById('classifyForm');
  const resultContainer = document.getElementById('resultContainer');
  
  if (form) {
    form.reset();
    updateCharCount.call(document.getElementById('issueDescription'));
  }
  
  if (resultContainer) {
    resultContainer.classList.add('hidden');
  }
  
  // Scroll to form
  form.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function downloadReport() {
  if (!state.currentResult) return;
  
  const report = generateReport(state.currentResult);
  const blob = new Blob([report], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `civic-issue-report-${Date.now()}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  showSuccess('Report downloaded successfully!');
}

function generateReport(result) {
  return `
CIVIC ISSUE URGENCY CLASSIFICATION REPORT
==========================================

Classification Results:
-----------------------
Urgency Level: ${result.urgency_level}
Urgency Score: ${result.urgency_score}/10
Confidence: ${(result.confidence * 100).toFixed(1)}%

Department Assignment:
----------------------
Recommended Department: ${result.recommended_department}
Estimated Response Time: ${result.estimated_response_time}

AI Analysis:
------------
${result.reasoning}

Technical Details:
------------------
Text Analysis Contribution: ${(result.text_contribution * 100).toFixed(0)}%
Image Analysis Contribution: ${(result.image_contribution * 100).toFixed(0)}%
Location Context: ${result.location_context}
Safety Context: ${result.safety_context}

Report Generated: ${new Date().toLocaleString()}
System: Civic Issue Urgency Classifier v1.0.0
  `.trim();
}

function showError(message) {
  showNotification(message, 'error');
}

function showSuccess(message) {
  showNotification(message, 'success');
}

function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `notification ${type} bounce-in`;
  notification.textContent = message;
  notification.style.cssText = `
    position: fixed;
    top: 100px;
    right: 20px;
    padding: 1rem 1.5rem;
    background: ${type === 'error' ? 'linear-gradient(135deg, #ff3b30, #ff8a00)' : 
                  type === 'success' ? 'linear-gradient(135deg, #34c759, #00d4aa)' : 
                  'linear-gradient(135deg, #667eea, #764ba2)'};
    color: white;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    z-index: 10000;
    max-width: 300px;
    font-weight: 600;
  `;
  
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.style.animation = 'slideOutRight 0.3s ease forwards';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// ============================================
//  ANIMATIONS
// ============================================
function initializeAnimations() {
  // Add stagger animation to feature cards
  const featureCards = document.querySelectorAll('.feature-card');
  featureCards.forEach((card, index) => {
    card.style.animationDelay = `${index * 0.1}s`;
    card.classList.add('stagger-item');
  });
}

// ============================================
//  SCROLL EFFECTS
// ============================================
function initializeScrollEffects() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
      }
    });
  }, { threshold: 0.1 });
  
  document.querySelectorAll('.scroll-reveal').forEach(el => {
    observer.observe(el);
  });
}

// ============================================
//  SMOOTH SCROLLING
// ============================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});

// Export functions for global use
window.classifyAnother = classifyAnother;
window.downloadReport = downloadReport;
