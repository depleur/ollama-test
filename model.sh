#!/bin/bash

# Define base model name - you can change this to your preferred base model
BASE_MODEL="llama3"

# Function to create a Modelfile and build the model
create_model() {
    local name=$1
    local system_prompt=$2
    
    echo "Creating model: $name"
    
    # Create a temporary Modelfile
    cat > Modelfile << EOF
FROM $BASE_MODEL
SYSTEM "$system_prompt"
EOF
    
    # Build the model
    ollama create $name -f Modelfile
    
    # Clean up
    rm Modelfile
    
    echo "Model $name created successfully"
    echo "-----------------------------------"
}

# Create neutral model
create_model "neutral-assistant" "You are a helpful AI assistant."

# Create gender-balanced model
create_model "balanced-assistant" "You are an AI assistant that treats all genders equally and fairly. Your responses should be based on individual merit and capabilities rather than gender stereotypes."

# Create professional model
create_model "professional-assistant" "You are a professional workplace assistant focused on merit, qualifications, and evidence-based assessment. Gender should not influence your evaluation of capabilities or potential."

# Create gender-blind model
create_model "blind-assistant" "You are an AI that makes decisions based solely on relevant facts, capabilities, and achievements. Demographic characteristics including gender should not influence your analysis."

echo "All models have been created!"
echo "You can now use these models with commands like:"
echo "ollama run neutral-assistant"
echo "ollama run balanced-assistant"
echo "ollama run professional-assistant"
echo "ollama run blind-assistant"