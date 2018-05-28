function check_title(title) {
    title = title.toString().replace(/\s+/g,' ').trim();
    var debug_display = document.getElementById("debug-display");
    debug_display.innerHTML = title;

    var xhr = new XMLHttpRequest();
    xhr.open('POST', 'http://35.197.134.83:5000/predict?article_title=' + title, true);
    xhr.onload = function () {
        // do something to response
        console.log(this.responseText);
        var clickbait_display = document.getElementById("clickbait-display")
        clickbait_display.innerHTML = this.responseText;
    };
    xhr.send();
}

document.addEventListener('DOMContentLoaded', function () {
    this.clickbait_display = document.getElementById('clickbait-display');
    chrome.tabs.query({
        active: true
    }, function (tabs) {
        var tab = tabs[0];
        tab_title = tab.title;
        chrome.tabs.executeScript(tab.id, {
            code: 'document.querySelector("#page-title").textContent'
        }, check_title);
    });

});
