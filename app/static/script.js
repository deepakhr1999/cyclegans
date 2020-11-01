async function bring(url){
    return new Promise((resolve, reject)=>{
        var http = new XMLHttpRequest()
        http.open("GET", url, true);
        http.setRequestHeader('Content-type', 'application/json');

        http.onreadystatechange = function() {
            try{
                if(http.readyState == 4 && http.status == 200) 
                    resolve(http.responseText)           
            }catch(err){
                reject(err)
            }
        }
        http.send()
    })
}

const refresh_images = ()=>{
    bring('/images')
    .then(text => {
        resp = JSON.parse(text)
        document.getElementById('imga').setAttribute('src', `data:image/png;base64,${resp.testA}`)
        document.getElementById('imgb').setAttribute('src', `data:image/png;base64,${resp.testB}`)
    })
}

console.log("GANS is fun!!")