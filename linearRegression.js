
//setup();
//mousePressed();
//draw();
var x_vals = [];
var y_vals = [];

let m, b;

const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate);

function setup(){
  createCanvas(400, 400);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function loss(pred, labels){
  return pred.sub(labels).square().mean();
}

function predict(x){
  const xs = tf.tensor1d(x);
  // y = mx + b
  const ys = xs.mul(m).add(b)
  return ys;
}

function mousePressed(){
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
  x_vals.push(x);
  y_vals.push(y);
}

function draw(){
  tf.tidy(() => {
  if (x_vals.length > 0){
    const ys = tf.tensor1d(y_vals);
    optimizer.minimize(() => loss(predict(x_vals), ys));
  }
  });

  background(0);
  stroke(255);
  strokeWeight(16);
  //if  ((x_vals.length != null)){
  //  console.log("DR1:",x_vals);
  //}
  //console.log("J",  x_vals.length)
  if (x_vals.length > 0){
    for (let i = 0; x_vals.length; i++) {
      if ((x_vals[i] == null)){break;}
      //console.log("DR2",i, x_vals[i], y_vals[i])
      let px = map(x_vals[i], 0, 1, 0, width);
      let py = map(y_vals[i], 0, 1, height, 0);
      //console.log("K",  px, py, x_vals[i], y_vals[i])
      point(Math.round(px), Math.round(py));
    }

    const xx = [0,1];
    const yy = predict(xx);
    yy.print();

    let x1 = map(xx[0], 0, 1,0, width);
    let x2 = map(xx[1], 0, 1,0, width);

    let liney = yy.dataSync();
    y1 =  map(liney[0], 0, 1,height, 0);
    y2 =  map(liney[1], 0, 1,height, 0);


    line(x1,y1,x2,y2);
    yy.dispose();

  }
}
