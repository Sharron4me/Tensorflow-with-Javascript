
  var xs= tf.randomUniform([20,1],0,150,'float32');
  //console.log(xs.print());
  var noise = tf.randomUniform([20,1],0,10 ,'float32');
  var targets = ((xs.mul(3)).add(9)).add(noise);
  console.log(targets.print());
  var c = document.getElementById("myCanvas");
  for(var i=0;i<7;i++){
    var x =  xs.slice([i, 0], 1).as1D().dataSync()[0];
    var y =  targets.slice([i, 0], 1).as1D().dataSync()[0];
    var ctx = c.getContext("2d");
    ctx.beginPath();
    ctx.arc(x,y,4,0,2*Math.PI);
    ctx.stroke();
    ctx.fillStyle = "Blue";
    ctx.fill();
    if(i<19){
      var x2 = xs.slice([i+1, 0], 1).as1D().dataSync()[0];
      var y2 = targets.slice([i+1, 0], 1).as1D().dataSync()[0];
      var ctx = c.getContext("2d");
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = "#02e5f9";
      ctx.stroke();
    }
  }

  var weights = tf.randomUniform([1,1],-0.1,0.1,'float32');
  var baises = tf.randomUniform([1],-0.1,0.1,'float32');
  var learning_rate =0.000017;
  var outputs;
  var delta;
  var loss;
  var deltas_scaled;
  for(var i=0;i<7;i++){
      outputs=(xs.dot(weights)).add(baises);
      delta = targets.sub(outputs);
      loss = ((outputs.squaredDifference(targets)).sum()).div(2).div(20);
      //console.log("Loss::"+loss);
      deltas_scaled = delta.div(20);
      // console.log("deltas sc: ");
      // console.log(deltas_scaled.reshape([20,1]).print());
      // console.log("XS:");
      // console.log(xs.transpose().reshape([1,20]).print());
      // console.log("xs shape:"+xs.shape);
      // console.log("deltasc shape:"+deltas_scaled.shape);
      weights = weights.sub(xs.transpose().reshape([1,20]).dot(deltas_scaled.reshape([20,1])).mul(learning_rate));
      baises = baises.sub(((deltas_scaled).sum()).mul(learning_rate));
  }
  console.log(outputs.print());
  for(var i=0;i<15;i++){
    var x =  xs.slice([i, 0], 1).as1D().dataSync()[0];
    var y =  outputs.slice([i, 0], 1).as1D().dataSync()[0];
    var ctx = c.getContext("2d");
    ctx.beginPath();
    ctx.arc(x,-1*y,4,0,2*Math.PI);
    ctx.stroke();
    ctx.fillStyle = "Red";
    ctx.fill();
    if(i<19){
      var x2 = xs.slice([i+1, 0], 1).as1D().dataSync()[0];
      var y2 = outputs.slice([i+1, 0], 1).as1D().dataSync()[0];
      var ctx = c.getContext("2d");
      ctx.beginPath();
      ctx.moveTo(x, -1*y);
      ctx.lineTo(x2, -1*y2);
      ctx.strokeStyle = "#fc0509";
      ctx.stroke();
    }
  }
  console.log(weights.print());
  var c = document.getElementById("prediction");
  var text = "Original weight (slope): 3.0 <br> Predicted weight : "+-1* weights.dataSync()[0] + "<br>";
  text+='<table class="prediction_table">';
  text+='<tr><th>Actual Value</th><th>Predicted Value</th></tr>';
  for(var i=0;i<15;i++){
    var y1 =  outputs.slice([i, 0], 1).as1D().dataSync()[0];
    var y2 =  targets.slice([i, 0], 1).as1D().dataSync()[0];
    text+='<tr><td>'+y2 + '</td><td>'+-1*y1+'</td></tr>';
  }
  text+='</table>';
  c.innerHTML = text;
