# Augmented Random Search on Brax Physics Engine (PyTorch)

Welcome to the PyTorch implementation of the Augmented Random Search (ARS) algorithm for Brax! Brax is a differentiable physics engine that simulates environments made up of rigid bodies, joints, and actuators. It is written in JAX and optimized for use on acceleration hardware, allowing for both efficient single-device simulation and scalable, massively parallel simulation on multiple devices.

This repository contains the PyTorch implementation of the ARS algorithm, which can be used to train policies for controlling the behavior of objects in a Brax simulated environment. The ARS algorithm is a model-free, gradient-free optimization method that has been shown to be effective at learning control policies in high-dimensional continuous action spaces.
