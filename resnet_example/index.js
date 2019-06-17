async function predict() {


  // Create an ONNX inference session with WebGL backend.
  const session = new onnx.InferenceSession({ backendHint: 'webgl' });

  // Load an ONNX model. This model is Resnet50 that takes a 1*3*224*224 image and classifies it.
  //await session.loadModel("./resnet.onnx");
  await session.loadModel("./emotion-model.onnx");

  

  // Load image.
  //const imageLoader = new ImageLoader(imageSize, imageSize);
  //const imageData = await imageLoader.getImageData('./happy.png');
  var canvas = document.getElementById('temp-canvas'),
  context = canvas.getContext('2d');
  base_image = new Image();
  base_image.src = './surprise.png';
  context.drawImage(base_image, 64, 64);


  // Preprocess the image data to match input dimension requirement, which is 1*3*224*224.
  const preprocessedData = preprocess_emotion(context, 64, 64);
  //const preprocessedData = preprocess(canvas, width, height);
  console.log(preprocessedData)
  
  //const resnetImage = new onnx.Tensor(preprocessedData, 'float32', [1, 1, width, height]);
  // Run model with Tensor inputs and get the result.
  //const outputMap = await session.run([resnetImage]);
  //const outputData = outputMap.values().next().value.data;

  
  const yolo_data = await session.run([preprocessedData]);
  console.log(yolo_data.values().next().value.data)
  const output = yolo_data.values().next().value.data;
    const emotionMap = ['neutral', 'happiness', 'surprise', 'sadness', 'anger',
       'disgust', 'fear', 'contempt'];
    const myOutput = softmax(output);
    let maxInd = -1;
    let maxProb = -1;
    for ( i = 0; i < myOutput.length; i++) {
      if (maxProb < myOutput[i]) {
        maxProb = myOutput[i];
        maxInd = i;
      }
    }

    console.log(emotionMap[maxInd])
  // Render the output result in html.
  printMatches(outputData);
}

function softmax(arr) {
    return arr.map(function(value,index) { 
      return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
    })
}
/**
 * Preprocess raw image data to match Resnet50 requirement.
 */
function preprocess(data, width, height) {
  const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
  const dataProcessed = ndarray(new Float32Array(width * height ), [1, 1, height, width]);

  // Normalize 0-255 to (-1)-1
  ndarray.ops.divseq(dataFromImage, 128.0);
  ndarray.ops.subseq(dataFromImage, 1.0);

  // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
  ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
  ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
  ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));
  
  return dataProcessed.data;
}

function scale (ctx){
    const scaledImage = document.getElementById('temp-canvas');
    const scaledCtx = scaledImage.getContext('2d');
    scaledImage.width = 64;
    scaledImage.height = 64;
    scaledCtx.drawImage(ctx.canvas, 64, 64);
    console.log(scaledCtx.getImageData(0, 0, 64, 64).data)
    return scaledCtx.getImageData(0, 0, 64, 64).data;
  }
function preprocess_emotion(ctx, width, height) {

    const data = this.scale(ctx);
    
    //const width = 64;
    //const height = 64;
    // data processing
    const greyScale = [];
    for (let i = 0; i < data.length; i+= 4) {
      greyScale.push((data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114 - 127.5)/127.5);
    }
    const tensor = new Tensor(new Float32Array(64*64), 'float32', [1, 1, 64, 64]);
    (tensor.data).set(greyScale);
    return tensor;
  }

/**
 * Utility function to post-process Resnet50 output. Find top k ImageNet classes with highest probability.
 */
function imagenetClassesTopK(classProbabilities, k) {
  if (!k) { k = 5; }
  const probs = Array.from(classProbabilities);
  const probsIndices = probs.map(
    function (prob, index) {
      return [prob, index];
    }
  );
  const sorted = probsIndices.sort(
    function (a, b) {
      if (a[0] < b[0]) {
        return -1;
      }
      if (a[0] > b[0]) {
        return 1;
      }
      return 0;
    }
  ).reverse();
  const topK = sorted.slice(0, k).map(function (probIndex) {
    const iClass = imagenetClasses[probIndex[1]];
    return {
      id: iClass[0],
      index: parseInt(probIndex[1], 10),
      name: iClass[1].replace(/_/g, ' '),
      probability: probIndex[0]
    };
  });
  return topK;
}

/**
 * Render Resnet50 output to Html.
 */
function printMatches(data) {
  let outputClasses = [];
  if (!data || data.length === 0) {
    const empty = [];
    for (let i = 0; i < 5; i++) {
      empty.push({ name: '-', probability: 0, index: 0 });
    }
    outputClasses = empty;
  } else {
    outputClasses = imagenetClassesTopK(data, 5);
  }
  const predictions = document.getElementById('predictions');
  predictions.innerHTML = '';
  const results = [];
  for (let i of [0, 1, 2, 3, 4]) {
    results.push(`${outputClasses[i].name}: ${Math.round(100 * outputClasses[i].probability)}%`);
  }
  predictions.innerHTML = results.join('<br/>');
}

