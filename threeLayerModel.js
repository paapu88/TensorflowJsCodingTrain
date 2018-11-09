
const model = tf.sequential();

const hidden = tf.layers.dense({
  units: 4,
  inputShape: [2],
  activation: 'sigmoid'
});


const output = tf.layers.dense({
  units: 1,
  activation: 'sigmoid'
});

model.add(hidden);
model.add(output);

const sgdOpt = tf.train.sgd(0.5);

const config = {
  optimizer: sgdOpt,
  loss: tf.losses.meanSquaredError
}

model.compile(config);

const xs = tf.tensor2d([
  [0,0],
  [0.5,0.5],
  [1,1]
]);

const ys = tf.tensor2d([
  [1],
  [0.5],
  [0]
]);

train().then(() => {
  console.log('training complete')
  let preds = model.predict(xs);
  preds.print();
})

async function train() {
  for (var i = 0; i < 100; i++) {
    const config ={shuffle: true, epochs:10}
    const response = await model.fit(xs, ys, config);
    console.log(response.history.loss[0]);
  }
}
