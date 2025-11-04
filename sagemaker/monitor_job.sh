#!/bin/bash

# Monitor SageMaker Training Job
# Usage: ./sagemaker/monitor_job.sh [JOB_NAME]

AWS_PROFILE="515422921637_AWSAdministratorAccess"
AWS_REGION="us-east-1"

if [ -z "$1" ]; then
    if [ -f /tmp/last_sagemaker_job.txt ]; then
        JOB_NAME=$(cat /tmp/last_sagemaker_job.txt)
        echo "Using last job: $JOB_NAME"
    else
        echo "Error: Please provide job name or run launch_sagemaker_job.sh first"
        exit 1
    fi
else
    # Extract job name from ARN if full ARN is provided
    if [[ "$1" == arn:aws:sagemaker:* ]]; then
        JOB_NAME=$(echo "$1" | sed 's/.*training-job\///')
        echo "Extracted job name from ARN: $JOB_NAME"
    else
        JOB_NAME=$1
    fi
fi

echo "Monitoring SageMaker job: $JOB_NAME"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    STATUS=$(aws sagemaker describe-training-job \
        --training-job-name "$JOB_NAME" \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --query 'TrainingJobStatus' \
        --output text 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "Error: Could not retrieve job status"
        exit 1
    fi

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    case $STATUS in
        "InProgress")
            echo "[$TIMESTAMP] Status: 🟡 In Progress - Training..."
            ;;
        "Completed")
            echo "[$TIMESTAMP] Status: ✅ Completed - Training finished successfully!"

            # Get output location
            if [ -f /tmp/last_sagemaker_bucket.txt ]; then
                S3_BUCKET=$(cat /tmp/last_sagemaker_bucket.txt)
                echo ""
                echo "Download your trained model:"
                echo "  aws s3 sync $S3_BUCKET/output/$JOB_NAME/output/ ./models/sagemaker_model/ --profile $AWS_PROFILE"
            fi
            exit 0
            ;;
        "Failed")
            echo "[$TIMESTAMP] Status: ❌ Failed - Training job failed"

            # Get failure reason
            REASON=$(aws sagemaker describe-training-job \
                --training-job-name "$JOB_NAME" \
                --profile "$AWS_PROFILE" \
                --region "$AWS_REGION" \
                --query 'FailureReason' \
                --output text)

            echo "Failure reason: $REASON"
            exit 1
            ;;
        "Stopped")
            echo "[$TIMESTAMP] Status: ⚠ Stopped - Training job was stopped"
            exit 1
            ;;
        *)
            echo "[$TIMESTAMP] Status: $STATUS"
            ;;
    esac

    # Wait 30 seconds before next check
    sleep 30
done
