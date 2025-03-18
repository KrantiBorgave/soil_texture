// App.js
import React, { useState, useEffect } from 'react';
import { View, Text, Button, Image, ActivityIndicator } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as ImagePicker from 'expo-image-picker';

// Load your TFLite model (place soil_model.tflite in assets/)
const modelJson = require('./assets/model.json');
const modelWeights = require('./assets/weights.bin');

export default function App() {
  const [model, setModel] = useState(null);
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);

  // Load TensorFlow and model
  useEffect(() => {
    (async () => {
      await tf.ready();
      const loadedModel = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
      setModel(loadedModel);
    })();
  }, []);

  // Pick image from gallery
  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      predictSoilType(result.assets[0].uri);
    }
  };

  // Predict soil type
  const predictSoilType = async (uri) => {
    if (!model) return;
    setLoading(true);
    try {
      // Preprocess image
      const imageTensor = await tf.reshape(
        tf.browser.fromPixels(await loadImage(uri)).resizeNearestNeighbor([224, 224]),
        [1, 224, 224, 3]
      );

      // Predict
      const output = model.predict(imageTensor);
      const predictedClass = tf.argMax(output.dataSync());
      setPrediction(`Soil Type: ${predictedClass}`);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={{ flex: 1, padding: 20 }}>
      <Button title="Upload Soil Image" onPress={pickImage} />
      {image && <Image source={{ uri: image }} style={{ width: 200, height: 200 }} />}
      {loading && <ActivityIndicator size="large" />}
      {prediction && <Text style={{ fontSize: 20 }}>{prediction}</Text>}
    </View>
  );
}

// Helper to load image
async function loadImage(uri) {
  const response = await fetch(uri);
  const blob = await response.blob();
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(blob);
  });
}