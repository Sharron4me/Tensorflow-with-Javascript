
  var xs= tf.randomUniform([20,1],0,100,'int32');
  console.log(xs.print());
  var noise = tf.randomUniform([20,1],0,2,'int32');
  var targets = ((xs.mul(3)).add(9)).add(noise);
  console.log(targets.print());
  var c = document.getElementById("myCanvas");
  for(var i=0;i<20;i++){
    var x =  xs.slice([i, 0], 1).as1D().dataSync()[0];
    var y =  targets.slice([i, 0], 1).as1D().dataSync()[0];
    var ctx = c.getContext("2d");
    ctx.beginPath();
    ctx.arc(x,y,4,0,2*Math.PI);
    ctx.stroke();
    ctx.fillStyle = "Blue";
    ctx.fill();
  }
