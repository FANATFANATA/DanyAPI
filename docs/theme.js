(function () {
    try {
        var t = localStorage.getItem('danyapi-theme');
        document.documentElement.setAttribute('data-theme', t || 'black');
    } catch (e) {
        document.documentElement.setAttribute('data-theme', 'black');
    }
})();
