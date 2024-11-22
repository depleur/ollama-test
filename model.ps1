# Define base model name - you can change this to your preferred base model
$BASE_MODEL = "llama3"

# Function to create a Modelfile and build the model
function Create-Model {
    param (
        [string]$Name,
        [string]$SystemPrompt
    )

    Write-Output "Creating model: $Name"

    # Create a temporary Modelfile
    $ModelfileContent = @"
FROM $BASE_MODEL
SYSTEM "$SystemPrompt"
"@

    $ModelfilePath = ".\Modelfile"
    Set-Content -Path $ModelfilePath -Value $ModelfileContent

    # Build the model
    & ollama create $Name -f $ModelfilePath

    # Clean up
    Remove-Item $ModelfilePath

    Write-Output "Model $Name created successfully"
    Write-Output "-----------------------------------"
}

# Create models
Create-Model "neutral-assistant" "You are a helpful AI assistant."
Create-Model "balanced-assistant" "You are an AI assistant that treats all genders equally and fairly. Your responses should be based on individual merit and capabilities rather than gender stereotypes."
Create-Model "professional-assistant" "You are a professional workplace assistant focused on merit, qualifications, and evidence-based assessment. Gender should not influence your evaluation of capabilities or potential."
Create-Model "blind-assistant" "You are an AI that makes decisions based solely on relevant facts, capabilities, and achievements. Demographic characteristics including gender should not influence your analysis."

Write-Output "All models have been created!"
Write-Output "You can now use these models with commands like:"
Write-Output "ollama run neutral-assistant"
Write-Output "ollama run balanced-assistant"
Write-Output "ollama run professional-assistant"
Write-Output "ollama run blind-assistant"