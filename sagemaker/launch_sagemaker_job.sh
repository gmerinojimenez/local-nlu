#!/bin/bash

# AWS SageMaker Training Job Launch Script
# Usage: ./sagemaker/launch_sagemaker_job.sh

set -e

# Configuration
AWS_PROFILE="515422921637_AWSAdministratorAccess"
AWS_REGION="us-east-1"  # Change if needed
SAGEMAKER_ROLE="arn:aws:iam::515422921637:role/MLPlatformInfra-Dev-UsEas-SageMakerExecutionRole784-19S6ZWNFZ08A"
S3_BUCKET="s3://sagemaker-nlu-training-$(date +%s)"  # Unique bucket name
JOB_NAME="nlu-t5-training-$(date +%Y%m%d-%H%M%S)"

# Instance configuration
INSTANCE_TYPE="ml.g4dn.xlarge"  # GPU instance (1 NVIDIA T4 GPU, 4 vCPUs, 16 GB RAM)
# Alternative options:
# - ml.p3.2xlarge (1 V100 GPU, faster but more expensive)
# - ml.m5.2xlarge (CPU only, cheaper but slower)
INSTANCE_COUNT=1
VOLUME_SIZE=50  # GB

echo "=========================================="
echo "SageMaker Training Job Configuration"
echo "=========================================="
echo "Job Name: $JOB_NAME"
echo "Instance: $INSTANCE_TYPE"
echo "S3 Bucket: $S3_BUCKET"
echo "AWS Profile: $AWS_PROFILE"
echo "=========================================="

# Step 1: Create S3 bucket
echo -e "\n[1/5] Creating S3 bucket..."
aws s3 mb "$S3_BUCKET" --profile "$AWS_PROFILE" --region "$AWS_REGION" || echo "Bucket may already exist"

# Step 2: Upload training data to S3
echo -e "\n[2/5] Uploading training data to S3..."
aws s3 sync data/processed/ "$S3_BUCKET/data/processed/" \
    --exclude "*" \
    --include "train.csv" \
    --include "val.csv" \
    --include "test.csv" \
    --profile "$AWS_PROFILE"

echo "Data uploaded to: $S3_BUCKET/data/processed/"

# Step 3: Package and upload source code
echo -e "\n[3/5] Packaging source code..."

# Create source package from current directory
tar -czf /tmp/nlu-source.tar.gz \
    src/ \
    sagemaker/train_sagemaker.py \
    sagemaker/requirements.txt \
    configs/ 2>/dev/null || echo "Warning: Some files may be missing"

# Verify tar file was created
if [ ! -f /tmp/nlu-source.tar.gz ]; then
    echo "Error: Failed to create source package"
    exit 1
fi

aws s3 cp /tmp/nlu-source.tar.gz "$S3_BUCKET/code/nlu-source.tar.gz" --profile "$AWS_PROFILE"
echo "Code uploaded to: $S3_BUCKET/code/"

# Step 4: Create training job configuration
echo -e "\n[4/5] Creating training job configuration..."
cat > /tmp/sagemaker-training-config.json <<EOF
{
    "TrainingJobName": "$JOB_NAME",
    "RoleArn": "$SAGEMAKER_ROLE",
    "AlgorithmSpecification": {
        "TrainingImage": "763104351884.dkr.ecr.$AWS_REGION.amazonaws.com/pytorch-training:2.0.1-gpu-py310",
        "TrainingInputMode": "File"
    },
    "InputDataConfig": [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "$S3_BUCKET/data/processed/",
                    "S3DataDistributionType": "FullyReplicated"
                }
            }
        }
    ],
    "OutputDataConfig": {
        "S3OutputPath": "$S3_BUCKET/output/"
    },
    "ResourceConfig": {
        "InstanceType": "$INSTANCE_TYPE",
        "InstanceCount": $INSTANCE_COUNT,
        "VolumeSizeInGB": $VOLUME_SIZE
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 86400
    },
    "HyperParameters": {
        "sagemaker_program": "train_sagemaker.py",
        "sagemaker_submit_directory": "$S3_BUCKET/code/nlu-source.tar.gz",
        "model-name": "t5-base",
        "num-epochs": "15",
        "batch-size": "16",
        "learning-rate": "0.00002"
    }
}
EOF

# Step 5: Launch training job
echo -e "\n[5/5] Launching SageMaker training job..."
aws sagemaker create-training-job \
    --cli-input-json file:///tmp/sagemaker-training-config.json \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION"

echo -e "\n=========================================="
echo "Training job launched successfully!"
echo "=========================================="
echo "Job Name: $JOB_NAME"
echo ""
echo "Monitor your job:"
echo "  aws sagemaker describe-training-job --training-job-name $JOB_NAME --profile $AWS_PROFILE"
echo ""
echo "View logs:"
echo "  aws sagemaker describe-training-job --training-job-name $JOB_NAME --profile $AWS_PROFILE --query 'TrainingJobStatus'"
echo ""
echo "Or use AWS Console:"
echo "  https://console.aws.amazon.com/sagemaker/home?region=$AWS_REGION#/jobs/$JOB_NAME"
echo ""
echo "When complete, download model:"
echo "  aws s3 sync $S3_BUCKET/output/$JOB_NAME/output/ ./models/sagemaker_model/ --profile $AWS_PROFILE"
echo "=========================================="

# Save job info for later
echo "$JOB_NAME" > /tmp/last_sagemaker_job.txt
echo "$S3_BUCKET" > /tmp/last_sagemaker_bucket.txt
