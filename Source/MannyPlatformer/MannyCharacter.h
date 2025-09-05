#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "MannyCharacter.generated.h"

UCLASS()
class MANNYPLATFORMER_API AMannyCharacter : public ACharacter
{
    GENERATED_BODY()

public:
    AMannyCharacter();

protected:
    virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

    void MoveForward(float Value);
    void MoveRight(float Value);
    void Turn(float Value);
    void LookUp(float Value);

private:
    UPROPERTY(VisibleAnywhere)
    class USpringArmComponent* CameraBoom;

    UPROPERTY(VisibleAnywhere)
    class UCameraComponent* FollowCamera;
};

