#include "GoalTrigger.h"
#include "Kismet/KismetSystemLibrary.h"
#include "Engine/Engine.h"
#include "MannyCharacter.h"

void AGoalTrigger::BeginPlay()
{
    Super::BeginPlay();

    OnActorBeginOverlap.AddDynamic(this, &AGoalTrigger::OnOverlapBegin);
}

void AGoalTrigger::OnOverlapBegin(AActor* OverlappedActor, AActor* OtherActor)
{
    if (!OtherActor)
    {
        return;
    }

    if (OtherActor->IsA(AMannyCharacter::StaticClass()))
    {
        if (GEngine)
        {
            GEngine->AddOnScreenDebugMessage(-1, 2.5f, FColor::Green, TEXT("You Win!"));
        }

        UKismetSystemLibrary::QuitGame(GetWorld(), nullptr, EQuitPreference::Quit, false);
    }
}

