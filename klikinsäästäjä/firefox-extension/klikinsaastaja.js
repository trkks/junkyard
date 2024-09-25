let links = document.querySelectorAll("a");
for (let link of links) {
    // Links on page might have query-params appended.
    if (link.href.startsWith("https://www.iltalehti.fi/treeni/a/a32d66ee-313d-42d7-b813-41400a84a4b6")) {
        let titleElem = link.querySelector(".front-title");
        let originalTitle = titleElem.textContent;
        let originalOp = titleElem.style.opacity;
        var timer;
        link.addEventListener("mouseenter", () => {
            console.log("enter");
            // Kudos:
            // https://stackoverflow.com/questions/6121203/how-to-do-fade-in-and-fade-out-with-javascript-and-css
            let op = 1;
            timer = setInterval(() => {
                console.log("fading out..");
                if (op <= 0.1) {
                    console.log("FADED OUT!");
                    clearInterval(timer);
                    titleElem.style.opacity = 0;
                    titleElem.textContent = "Selkämakkaroista pääsee eroon tekemällä selkälihasliikkeitä";
                    timer = setInterval(() => {
                        console.log("fading in..");
                        if (op >= 0.9) {
                            console.log("FADED IN!");
                            clearInterval(timer);
                            titleElem.style.opacity = 1;
                        } else {
                            titleElem.style.opacity = op;
                            op += op * 1.1;
                        }
                    }, 50);
                } else {
                    titleElem.style.opacity = op;
                    op -= op * 0.1;
                }
            }, 10);
        });
        link.addEventListener("mouseleave", () => {
            console.log("leave");
            clearInterval(timer);
            titleElem.textContent = originalTitle;
        });
        break;
    }
}
