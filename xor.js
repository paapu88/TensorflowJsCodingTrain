
let nn
let myModel
let resolution = 40
let cols, rows
let xs
let y_values

let train_xs = tf.tensor2d([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

let train_ys =tf.tensor2d([
    [0],
    [1],
    [1],
    [0]
])

function setup(){
  createCanvas(400,400);
  cols = width/resolution;
  rows = height/resolution;
  let inputs = []
  for (var i = 0; i<cols; i++) {
    for (var j = 0; j<rows; j++) {
      let x1 = i/cols
      let x2 = j/rows
      inputs.push([x1,x2])
    }
  }
  xs = tf.tensor2d(inputs)

  //nn = new NeuralNetwork(2,2,1)
  myModel = tf.sequential()
  let hidden = tf.layers.dense({
    inputShape: [2],
    units:4,
    activation:'sigmoid'
  })
  let output =tf.layers.dense({
    units:1,
    activation:'sigmoid'
  })
  myModel.add(hidden)
  myModel.add(output)

  myModel.compile({
    optimizer: tf.train.adam(0.1),
    loss: tf.losses.meanSquaredError
  })
  setTimeout(train, 100)
}

function train(){
  trainModel().then(result => {
    console.log(result.history.loss[0])
    setTimeout(train, 100)
  })
}


function trainModel(){
  return myModel.fit(train_xs, train_ys,
    {shuffle:true, epochs:2})
}

function draw(){
  background(0);

  ys = myModel.predict(xs)
  y_values = ys.dataSync()
  //draw
  let index = 0
  if (y_values[0] != null){
  for (var i = 0; i<cols; i++) {
    for (var j = 0; j<rows; j++) {
      let br = y_values[index]*255
      fill(br)
      rect(i*resolution, j*resolution,resolution, resolution)
      fill(255-br)
      textAlign(CENTER, CENTER)
      text(nf(y_values[index],1,2),i*resolution+resolution/2,
       j*resolution+resolution/2)
      index++
    }}}
}
