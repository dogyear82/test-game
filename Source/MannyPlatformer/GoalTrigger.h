#pragma once

#include "CoreMinimal.h"
#include "Engine/TriggerBox.h"
#include "GoalTrigger.generated.h"

UCLASS()
class MANNYPLATFORMER_API AGoalTrigger : public ATriggerBox
{
    GENERATED_BODY()

protected:
    virtual void BeginPlay() override;

    UFUNCTION()
    void OnOverlapBegin(AActor* OverlappedActor, AActor* OtherActor);
};

