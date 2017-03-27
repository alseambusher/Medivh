var heatmap;
    document.querySelectorAll("img").forEach((img) => {
        if (img.currentSrc)
            img.src = "http://localhost:8000/image?path=" + img.currentSrc;
    });

setTimeout(() => {
    var request = new XMLHttpRequest();
    request.open('GET', 'http://localhost:8001', true);

    request.onload = function() {
        if (request.status >= 200 && request.status < 400) {
            var data = JSON.parse(request.responseText);

            heatmap = h337.create({
                container: document.body
            });
            heatmap.setData({
                max: 3,
                data: data.results
//                data: [{ x: 10, y: 105, value: 5}]
            });
        }
    };

    request.send();
}, 1000);

