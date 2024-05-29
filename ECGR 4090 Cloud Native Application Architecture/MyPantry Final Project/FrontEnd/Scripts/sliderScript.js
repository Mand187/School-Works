// Select the dark mode toggle switch
const darkModeToggle = document.getElementById('darkModeToggle');

// Listen for changes to the toggle switch
darkModeToggle.addEventListener('change', () => {
  // Check if the toggle switch is checked
  if (darkModeToggle.checked) {
    // If checked, enable dark mode by adding a class to the body
    document.body.classList.add('dark-mode');
  } else {
    // If not checked, disable dark mode by removing the class from the body
    document.body.classList.remove('dark-mode');
  }
});
