/*************************************************************************
    OpenGD - Open source Geometry Dash.
    Copyright (C) 2023  OpenGD Team

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License    
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*************************************************************************/

#include "EffectManager.h"

EffectManager* EffectManager::create()
{
	EffectManager* ret = new EffectManager();
	if (ret->init())
	{
		ret->autorelease();
		return ret;
	}
	else
	{
		delete ret;
		ret = nullptr;
		return nullptr;
	}
}

void EffectManager::runMoveCommand(float duration, ax::Point offsetPos, int easeType, float easeAmt, int groupID)
{
	GroupCommandObject* gco = GroupCommandObject::create();
	gco->retain();
	gco->_groupID = groupID;
	gco->runMoveCommand(duration, offsetPos, easeType, easeAmt);
	gco->_actionID = 0;
	_groupActions.push_back(gco);
}

void EffectManager::prepareMoveActions(float dt, bool idk)
{
	for(auto d : _activeMoveActions) d.second->release();
	_activeMoveActions.clear();
	for (auto commandObject : _groupActions)
	{
		if (commandObject->_actionID == 0)
		{
			CCMoveNode* moveNode;
			if (!this->_activeMoveActions.contains(commandObject->_groupID))
			{
				moveNode = CCMoveNode::create();
				moveNode->retain();
				this->_activeMoveActions[commandObject->_groupID] = moveNode;
			}
			if (!moveNode)
				moveNode = this->_activeMoveActions[commandObject->_groupID];

			ax::Point unk, newPos;

			if (!commandObject->_actionDoneForNextLoop)
			{
				commandObject->step(dt);
				moveNode->_delta1 = commandObject->_delta1;
				commandObject->_delta2 += commandObject->_delta1;

				unk = commandObject->_unkPoint, newPos = commandObject->_newPos;

				if (commandObject->_followPlayerX)
				{
					unk.x = this->_xAccel;
					if (idk)
						commandObject->_newPos.x += this->_xAccel;
					else
						newPos.x += this->_xAccel;
				}
				if (commandObject->_followPlayerY)
				{
					unk.y = this->_yAccel;
					if (idk)
						commandObject->_newPos.y += this->_yAccel;
					else
						newPos.y += this->_yAccel;
				}

				moveNode->_newPosStatic.x += unk.x;
				moveNode->_newPosStatic.y += unk.y;
				moveNode->_newPosOptimized.x += newPos.x;
				moveNode->_newPosOptimized.y += newPos.y;
			}
			else
			{
				moveNode->_newPosOptimized.x += newPos.x;
				moveNode->_newPosOptimized.y += newPos.y;
			}

			commandObject->_delta1 = 0;

			if(commandObject->_actionDoneForNextLoop)
				this->_tempCompletedActions.push_back(commandObject);
			else if(commandObject->_actionDone)
				commandObject->_actionDoneForNextLoop = true;
		}
	}

	for(auto ac : this->_tempCompletedActions)
	{
		this->_completedMoveActions.push_back(ac);
		this->_groupActions.erase(std::remove(this->_groupActions.begin(), this->_groupActions.end(), ac), this->_groupActions.end());
	}
	this->_tempCompletedActions.clear();
}