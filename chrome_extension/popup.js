function check_title(title) {
    title = title.toString().replace(/\s+/g, ' ').trim();
    var title_display = document.getElementById("title-display");
    var clickbait_display = document.getElementById("clickbait-display")
    var debug_display = document.getElementById("debug-display")

    title_display.innerHTML = title.toString();

    var xhr = new XMLHttpRequest();
    xhr.open('POST', 'http://35.185.181.66:5000/predict?article_title=' + title, true);
    xhr.onload = function () {
        // do something to response
        var json = JSON.parse(this.responseText);
        console.log(json);

        clickbait_display.innerHTML = json.clickbait;

    };
    xhr.send();
}

function check_img(image_list) {
    image_list = image_list[0].toString();
    image_list = image_list.substring(1);
    
    var xhr = new XMLHttpRequest();
    xhr.open('POST', 'http://35.185.181.66:5000/predict?image_list=' + image_list, true);
    xhr.onload = function () {
        // do something to response
        var json = JSON.parse(this.responseText);
        console.log(json);

        //clickbait_display.innerHTML = json.img_preds;

    };
    xhr.send();
}

function extract_hostname(url) {
    var a = document.createElement('a');
    a.href = url;
    return a.hostname;
}

function extract_img_sources() {
    var imgs = document.getElementsByTagName("img");var img_sources = [];for (var i = 0; i < imgs.length; i++) {img_sources.push(imgs[i].src);}
    return img_sources;
}

document.addEventListener('DOMContentLoaded', function () {
    this.clickbait_display = document.getElementById('clickbait-display');
    chrome.tabs.query({
        active: true
    }, function (tabs) {
        var tab = tabs[0];
        var page_domain = extract_hostname(tab.url.toString());
        var debug_display = document.getElementById("debug-display")
        debug_display.innerHTML = page_domain;

        chrome.tabs.executeScript(tab.id, {
            code: 'var imgs = document.getElementsByTagName("img");var img_sources = [];for (var i = 0; i < imgs.length; i++) {img_sources.push(imgs[i].src);img_sources;}'
        }, check_img);

        switch (page_domain) {
            case "www.allsingaporestuff.com":
                chrome.tabs.executeScript(tab.id, {
                    code: 'document.querySelector("#page-title").textContent'
                }, check_title);
                break;

            case "www.straitstimes.com":
                chrome.tabs.executeScript(tab.id, {
                    code: 'document.querySelector(".headline").textContent'
                }, check_title);
                break;

            case "www.channelnewsasia.com":
                chrome.tabs.executeScript(tab.id, {
                    code: 'document.querySelector(".article__title").textContent'
                }, check_title);

            case "www.buzzfeed.com":
                chrome.tabs.executeScript(tab.id, {
                    code: 'document.querySelector(".buzz-title").textContent'
                }, check_title);
                break;

            default:
                var tab_title = tab.title;
                check_title(tab_title);
        }

    });

});
