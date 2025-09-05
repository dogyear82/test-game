#include "MannyCharacter.h"

#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/SpringArmComponent.h"
#include "UObject/ConstructorHelpers.h"
#include "Components/SkeletalMeshComponent.h"

AMannyCharacter::AMannyCharacter()
{
    // Size for collision capsule
    GetCapsuleComponent()->InitCapsuleSize(42.f, 96.0f);

    // Do not rotate character with controller; orient via movement
    bUseControllerRotationPitch = false;
    bUseControllerRotationYaw = false;
    bUseControllerRotationRoll = false;
    GetCharacterMovement()->bOrientRotationToMovement = true;
    GetCharacterMovement()->RotationRate = FRotator(0.f, 540.f, 0.f);
    GetCharacterMovement()->JumpZVelocity = 600.f;
    GetCharacterMovement()->AirControl = 0.3f;

    // Create a camera boom (pulls in towards the character if there is a collision)
    CameraBoom = CreateDefaultSubobject<USpringArmComponent>(TEXT("CameraBoom"));
    CameraBoom->SetupAttachment(RootComponent);
    CameraBoom->TargetArmLength = 300.0f; // The camera follows at this distance behind the character
    CameraBoom->bUsePawnControlRotation = true; // Rotate the arm based on the controller

    // Create a follow camera
    FollowCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("FollowCamera"));
    FollowCamera->SetupAttachment(CameraBoom, USpringArmComponent::SocketName);
    FollowCamera->bUsePawnControlRotation = false; // Camera does not rotate relative to arm

    // Load Manny skeletal mesh from Engine content
    static ConstructorHelpers::FObjectFinder<USkeletalMesh> MannyMesh(TEXT("SkeletalMesh'/Engine/Characters/Mannequins/Meshes/SKM_Manny.SKM_Manny'"));
    if (MannyMesh.Succeeded())
    {
        GetMesh()->SetSkeletalMesh(MannyMesh.Object);
        // Match the third-person template offsets
        GetMesh()->SetRelativeLocation(FVector(0.f, 0.f, -90.f));
        GetMesh()->SetRelativeRotation(FRotator(0.f, -90.f, 0.f));
    }

    // Without an animation blueprint, the mesh will be static (ok for minimal demo)
    GetMesh()->SetAnimationMode(EAnimationMode::AnimationSingleNode);
}

void AMannyCharacter::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
    check(PlayerInputComponent);
    PlayerInputComponent->BindAxis("MoveForward", this, &AMannyCharacter::MoveForward);
    PlayerInputComponent->BindAxis("MoveRight", this, &AMannyCharacter::MoveRight);
    PlayerInputComponent->BindAxis("Turn", this, &AMannyCharacter::Turn);
    PlayerInputComponent->BindAxis("LookUp", this, &AMannyCharacter::LookUp);
    PlayerInputComponent->BindAction("Jump", IE_Pressed, this, &ACharacter::Jump);
    PlayerInputComponent->BindAction("Jump", IE_Released, this, &ACharacter::StopJumping);
}

void AMannyCharacter::MoveForward(float Value)
{
    if (Controller && Value != 0.0f)
    {
        const FRotator YawRot(0.f, Controller->GetControlRotation().Yaw, 0.f);
        const FVector Direction = FRotationMatrix(YawRot).GetUnitAxis(EAxis::X);
        AddMovementInput(Direction, Value);
    }
}

void AMannyCharacter::MoveRight(float Value)
{
    if (Controller && Value != 0.0f)
    {
        const FRotator YawRot(0.f, Controller->GetControlRotation().Yaw, 0.f);
        const FVector Direction = FRotationMatrix(YawRot).GetUnitAxis(EAxis::Y);
        AddMovementInput(Direction, Value);
    }
}

void AMannyCharacter::Turn(float Value)
{
    AddControllerYawInput(Value);
}

void AMannyCharacter::LookUp(float Value)
{
    AddControllerPitchInput(Value);
}

