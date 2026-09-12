<!-- Rendered version: https://claude.ai/code/artifact/f5290de5-216c-40bc-8316-576b8604e235 -->

# Fargate Deployment Ledger

Account 268666185034 (mohitSinghEctech) · Region ap-south-1 (Mumbai) · 12 September 2026
Service: api-task (FARGATE) · Image: backend:v2 · Credits: $120.00 expiring 2027-09-11

A containerized FastAPI service (grounded_answer_engine) built, pushed to ECR, and run
on ECS Fargate. Record of every resource created, what each bills, and the controls that
hold the idle cost at half a cent a month.

## Status

  Meter                 STOPPED (desired 0, running 0)
  Idle cost             $0.005 / month   (ECR storage only)
  Cost when running     $0.46 / day      (0.25 vCPU / 0.5 GB)
  Spend to date         $0.00 of $10.00 budget

## What was built

  Dockerfile                ECR                   ECS Fargate          Upstream
  python:3.12-alpine   ->   backend           ->  firstCluster    ->   gemini-3.7-flash
  multi-stage               v1, v2 · amd64        / api-task           via public IP
  non-root                  50.69 MB stored       256 CPU · 512 MB     no NAT gateway

A single stateless container. No database, no vector store, no disk state - which is why
it scales cleanly to zero.

LLM_API_KEY is resolved at container start from SSM Parameter Store. Never in the image,
never in git.

Task definition revisions:
  :1  broken (no image tag, curl health check, 1 vCPU / 3 GB)
  :2  backend:v1 -> version 0.1.0
  :3  backend:v2 -> version 0.2.0   (currently selected)

## The ledger

| Resource                  | What it is                                          | Rate            | Meter      |
|---------------------------|-----------------------------------------------------|-----------------|------------|
| ECR repo backend          | 2 images, 12 unique blobs, dedup to 50.69 MB        | $0.10 / GB-mo   | $0.0049/mo |
| ECS cluster firstCluster  | Namespace only. Container Insights disabled.        | -               | Free       |
| ECS service api-task      | Supervisor config, desired count 0                  | -               | Free       |
| Task definitions x3       | api-task:1, :2, :3 - immutable revision history     | -               | Free       |
| Fargate tasks             | 0.25 vCPU + 0.5 GB. Per-second billing.             | $0.0142 / hr    | Stopped    |
| Public IPv4               | On the task ENI. Exists only while a task runs.     | $0.005 / hr     | Stopped    |
| ecsTaskExecutionRole      | ECR pull, CloudWatch, ssm:GetParameters /gae/prod/* | -               | Free       |
| ecsInstanceRole           | Built for EC2 launch type, unused                   | -               | Free       |
| AWSServiceRoleForECS      | AWS-managed service-linked role                     | -               | Free       |
| GaeCliAccess policy       | Scoped CLI perms; no role/user/policy creation      | -               | Free       |
| /gae/prod/llm-api-key     | SecureString, standard tier, aws/ssm KMS key        | -               | Free       |
| /ecs/api-task             | 28.5 KB stored, 7-day retention                     | 5 GB free       | Free       |
| gae-task-sg               | One inbound rule: tcp/8000. Dedicated SG.           | -               | Free       |
| Monthly cost budget       | $10 threshold with email alert                      | 2 free          | Free       |

  IDLE TOTAL                $0.005 / month
  WITH ONE TASK RUNNING     $0.46 / day

## Billing and not billing

ECR bills STORAGE - bytes at rest, charged whether or not anything runs.
ECS bills USAGE - task-seconds only. No reserved capacity, no minimum, no idle fee.
That is what makes scale-to-zero genuinely zero.

  Accruing right now:
    ECR storage, 50.69 MB ............ $0.005/mo   (the entire list - six cents a year)

  Dormant until a task starts:
    Fargate compute .................. $0.34/day
    Public IPv4 address .............. $0.12/day
    CloudWatch log ingest ............ within 5 GB free

## Removed, and never built

| Resource              | Why                                                        | Avoided     | Status        |
|-----------------------|------------------------------------------------------------|-------------|---------------|
| Application Load Bal. | Created before needed. One task on its own IP needs none.  | $17.50/mo   | Deleted       |
| Elastic IP            | Left over from the ALB. All public IPv4 billed since 2024. | $3.65/mo    | Released      |
| Container Insights    | On by default. ~16-24 CloudWatch metrics at $0.30 each.    | ~$6/mo      | Disabled      |
| NAT Gateway           | Tutorials use private subnets. A public IP does the job.   | $32/mo      | Never created |
| EC2 container inst.   | On a credit account t3.micro costs the same as Fargate.    | $14/mo      | Never created |

  RECURRING COST AVOIDED    $73.15 / month

## The seven guards

1. A $10 monthly budget with an email alert
   The backstop for everything else. Free for the first two budgets, two minutes to
   create, and it earned a $20 AWS credit on top.

2. Desired count 0 between sessions                                   -$0.46/day
   Fargate bills per second for running tasks only. Scaling to zero stops both compute
   and the IPv4 charge while every configuration object survives untouched.

3. Task sized to 0.25 vCPU / 0.5 GB                                   -$80/mo
   The console's first task definition asked for 1 vCPU and 3 GB, twice over.
   Right-sizing cut the per-task rate from $0.0619/hr to $0.0142/hr.

4. Container Insights off                                             -$6/mo
   On by default for console-created clusters. Bills as CloudWatch custom metrics,
   against a free allowance of only ten.

5. Public IP instead of a NAT gateway                                 -$28/mo
   The task needs outbound access to ECR, SSM, CloudWatch and Gemini. A public IP
   provides it for $3.65/mo against a NAT gateway's $32, which bills hourly just to exist.

6. ECR lifecycle policy on untagged images
   Every re-push of a mutable tag orphans the previous image, which keeps consuming
   storage silently. The policy expires untagged images after one day.

7. Seven-day log retention
   CloudWatch log groups default to never expire. Irrelevant at 28 KB, material on a
   chatty service left running for a year.

## Incidents

1. An admin access key was exposed in plaintext            ROTATED, ~3 hr exposure
   A long-lived key for mohitsingh-cli carrying AdministratorAccess was pasted into a
   chat transcript. A replacement key was issued, the CLI reconfigured, the original
   deleted, and the duplicate copy written to root's config removed. Leaked keys with
   admin rights are actively scanned for.
   Follow-up: the user was scoped down from AdministratorAccess to a custom
   GaeCliAccess policy, verified by confirming iam:CreateUser and iam:AttachUserPolicy
   are now refused.

2. A service configured for two 1-vCPU tasks               CAUGHT BEFORE RUNNING
   Would have cost roughly $90/month had the image pull succeeded. The failing pull was,
   accidentally, the thing that prevented it.

3. Image pushed as arm64 from an Apple Silicon Mac         FIXED
   An arm64 image cannot run on x86 infrastructure. Rebuilt with --platform linux/amd64,
   and the architecture is now verified before every push.

4. Task definition referenced an image with no tag         FIXED
   Docker resolved it to :latest, which did not exist in the repository. Pinning explicit
   tags is also what makes rollback meaningful.

5. Health check called curl, which Alpine doesn't ship     FIXED
   Would have failed every probe, producing an endless kill-and-restart loop that bills
   for each attempt. Replaced with BusyBox wget.

6. Service permanently bound to a deleted target group     RECREATED
   An ECS service's load balancer configuration is fixed at creation and cannot be
   removed. The service was deleted and recreated - free, and it touches nothing else.

7. Load balancer with no inbound rule                      SUPERSEDED
   It shared the VPC default security group, whose only rule allowed traffic from itself.
   Nothing on the internet could reach it.

8. Cluster creation raced the service-linked role          RETRIED
   The first ECS action in any account must create AWSServiceRoleForECS first, and IAM is
   eventually consistent. Retrying succeeded.

## Runbook

All commands assume ap-south-1. The service comes back on api-task:3, serving 0.2.0.

Start serving (~60 seconds to a running task on a fresh public IP):

    aws ecs update-service --cluster firstCluster --service api-task \
      --desired-count 1 --region ap-south-1

Find the public IP (it changes on every deployment - that's a load balancer's job):

    TASK=$(aws ecs list-tasks --cluster firstCluster --region ap-south-1 \
      --query 'taskArns[0]' --output text)
    ENI=$(aws ecs describe-tasks --cluster firstCluster --tasks $TASK --region ap-south-1 \
      --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value|[0]' \
      --output text)
    aws ec2 describe-network-interfaces --network-interface-ids $ENI --region ap-south-1 \
      --query 'NetworkInterfaces[0].Association.PublicIp' --output text

Stop the meter (the one command that matters; everything else survives at no cost):

    aws ecs update-service --cluster firstCluster --service api-task \
      --desired-count 0 --region ap-south-1

Roll back to 0.1.0 (no rebuild, no git, no push - revision 2 points at backend:v1):

    aws ecs update-service --cluster firstCluster --service api-task \
      --task-definition api-task:2 --region ap-south-1

Audit for anything billable (empty output is the correct answer):

    aws ec2 describe-instances --region ap-south-1 \
      --query 'Reservations[].Instances[?State.Name!=`terminated`].InstanceId' --output text
    aws elbv2 describe-load-balancers --region ap-south-1 \
      --query 'LoadBalancers[].LoadBalancerName' --output text
    aws ec2 describe-addresses --region ap-south-1 --query 'Addresses[].PublicIp' --output text
    aws ec2 describe-nat-gateways --region ap-south-1 \
      --query 'NatGateways[?State==`available`].NatGatewayId' --output text
    aws rds describe-db-instances --region ap-south-1 \
      --query 'DBInstances[].DBInstanceIdentifier' --output text

Export the CLI user's permissions (AWS holds the authoritative copy, so there is no
policy file to keep in sync):

    aws iam get-policy-version \
      --policy-arn arn:aws:iam::268666185034:policy/GaeCliAccess \
      --version-id $(aws iam get-policy \
        --policy-arn arn:aws:iam::268666185034:policy/GaeCliAccess \
        --query 'Policy.DefaultVersionId' --output text) \
      --query 'PolicyVersion.Document' > cli-policy.json

Check spend without paying to check (Cost Explorer's API charges $0.01 per request;
the Budgets API is free, and so is the Cost Explorer console):

    aws budgets describe-budgets --account-id 268666185034 \
      --query 'Budgets[].{limit:BudgetLimit.Amount,spend:CalculatedSpend.ActualSpend.Amount}'

## Rate card (ap-south-1 list prices)

  Fargate vCPU ......................... $0.04656 / hr
  Fargate memory ....................... $0.00511 / GB-hr
  Public IPv4 .......................... $0.005 / hr        <- bills hourly regardless
  Application Load Balancer ............ $0.0225 / hr + LCU <- bills hourly regardless
  NAT Gateway .......................... $0.045 / hr + data <- bills hourly regardless
  EC2 t3.micro ......................... $0.0108 / hr       <- bills hourly regardless
  ECR storage .......................... $0.10 / GB-month
  CloudWatch custom metric ............. $0.30 / metric-month
  ECR pull to same region .............. free
  ECS control plane .................... free
  IAM, Parameter Store (standard) ...... free
  Data transfer out .................... 100 GB/month free

## Open items

1. POST /ask has no authentication or rate limit          BEFORE GOING PUBLIC
   Every call spends Gemini quota. Today it's protected only by being an unmemorable IP
   on a non-standard port, and by the service being stopped. An API-key check in
   middleware is the minimum before attaching a domain name.

2. The CLI key is scoped now, but still never expires      PARTLY DONE
   Permissions are down from AdministratorAccess to GaeCliAccess (v2), so a leak costs
   this one deployment rather than the whole account and its credits. What remains is
   the key's lifetime: it has no expiry, so a leak is permanent until noticed.
   Short-lived credentials via 'aws login' or IAM Identity Center close that.

3. The image carries 58 MB it doesn't need                OPTIONAL
   Development dependencies (ruff alone is 21 MB), precompiled bytecode, and unstripped
   debug symbols in the source-compiled uvloop extension. Trimming takes 160 MB of
   filesystem down to roughly 100 MB.

---
Figures measured directly from the account on 12 September 2026, not estimated.
ECR storage computed from unique layer digests across both image manifests.
AWS list prices for ap-south-1.
Credits: $100 AWS Free Tier + $20 Budgets activity, both expiring 2027-09-11.
