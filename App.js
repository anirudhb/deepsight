import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient'
import scale from './utils/scale'

export default function App() {
  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#2c51c9', '#213b8f']}
        style={styles.gradient}>
        <View
          style={styles.card}>
            <Text>Deep Sight</Text>
        </View>
      </LinearGradient>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#2c51c9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  gradient: {
    width: scale(375, 0),
    height: scale(705, 1),
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    backgroundColor: 'white',
    height: scale(200,0),
    width: scale(250,1),
    borderRadius: 20
  }
});
