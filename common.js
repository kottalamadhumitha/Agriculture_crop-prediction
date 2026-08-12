function capitalize(word) {
    if (!word) return "";
    return word.charAt(0).toUpperCase() + word.slice(1);
}

function showToast(message) {
    const toast = document.createElement("div");
    toast.textContent = message;
    toast.className = "toast-msg";
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3200);
}
