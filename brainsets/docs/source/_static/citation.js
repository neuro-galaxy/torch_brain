document.addEventListener('DOMContentLoaded', function () {
    // Handle cite button clicks
    document.querySelectorAll('.cite-button').forEach(button => {
        button.addEventListener('click', function (event) {
            event.stopPropagation(); // Prevent event from bubbling up
            const doi = this.getAttribute('data-doi');
            const popup = document.getElementById(`popup-${doi}`);

            // Toggle visibility
            if (popup.style.display === 'none') {
                // Close any other open popups first
                document.querySelectorAll('.citation-popup').forEach(p => {
                    p.style.display = 'none';
                });
                popup.style.display = 'block';
            } else {
                popup.style.display = 'none';
            }
        });
    });

    // Handle copy buttons
    document.querySelectorAll('.copy-button').forEach(button => {
        button.addEventListener('click', function (event) {
            event.stopPropagation(); // Prevent event from bubbling up

            // Get the content element (sibling of the button)
            const contentContainer = this.closest('.citation-content');
            // Find either the pre or p element inside
            const contentElement = contentContainer.querySelector('pre') || contentContainer.querySelector('p');

            // Copy to clipboard
            navigator.clipboard.writeText(contentElement.textContent).then(() => {
                // Show success feedback
                this.innerHTML = '<i class="fa-solid fa-check"></i>';
                setTimeout(() => {
                    this.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round"><title>Copy to clipboard</title><path stroke="none" d="M0 0h24v24H0z" fill="none"></path><rect x="8" y="8" width="12" height="12" rx="2"></rect><path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path></svg>';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
            });
        });
    });

    // Close popups when clicking outside
    document.addEventListener('click', function (event) {
        if (!event.target.closest('.citation-popup') && !event.target.closest('.cite-button')) {
            document.querySelectorAll('.citation-popup').forEach(popup => {
                popup.style.display = 'none';
            });
        }
    });

    // Prevent popup from closing when clicking inside it
    document.querySelectorAll('.citation-popup').forEach(popup => {
        popup.addEventListener('click', function (event) {
            event.stopPropagation(); // Prevent event from bubbling up
        });
    });
});
