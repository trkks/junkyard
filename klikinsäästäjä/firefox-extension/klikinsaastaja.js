const main = () => {
    let links = document.querySelectorAll("a");
    for (let link of links) {
        let titleElem = link.querySelector(".front-title");
        if (!titleElem) { continue; }
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
                    titleElem.textContent = "FAKE NEWZ";
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
            titleElem.style.opacity = 1;
            titleElem.textContent = originalTitle;
        });
    }
};

window.onload = () => setTimeout(main, 750);
